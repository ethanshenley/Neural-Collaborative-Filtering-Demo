import torch
import numpy as np
from typing import Dict, List, Union, Optional
from collections import defaultdict
from sklearn import metrics

import logging

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    batch_size: Optional[int] = None
) -> Dict[str, float]:
    """Calculate all recommendation metrics comprehensively.
    
    Args:
        predictions: Model prediction scores (N, 1)
        targets: Ground truth values (N, 1)
        k_values: List of k values for cut-off metrics
        batch_size: Original batch size for reshaping (before negative sampling)
        
    Returns:
        Dictionary containing all metrics
    """
    try:
        # Ensure tensors are on CPU and correct shape
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()
        
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
            
        # If batch_size not provided, try to infer it
        if batch_size is None:
            # Assume first occurrence of target 1 starts a new batch
            target_ones = (targets == 1).nonzero().squeeze()
            if len(target_ones) > 1:
                batch_size = target_ones[1] - target_ones[0]
            else:
                batch_size = len(targets)
        
        # Initialize metrics dictionary
        metrics = defaultdict(float)
        
        # Calculate metrics for each k
        for k in k_values:
            # Basic metrics
            metrics[f'hit_rate@{k}'] = calculate_hit_rate(predictions, targets, k, batch_size)
            metrics[f'ndcg@{k}'] = calculate_ndcg(predictions, targets, k, batch_size)
            metrics[f'mrr@{k}'] = calculate_mrr(predictions, targets, k, batch_size)
            metrics[f'map@{k}'] = calculate_map(predictions, targets, k, batch_size)
        
        # Additional metrics
        metrics['auc'] = calculate_auc(predictions, targets)
        metrics['accuracy'] = calculate_accuracy(predictions, targets)
        
        # Separate positive/negative metrics
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        if pos_mask.any():
            metrics['pos_accuracy'] = calculate_accuracy(predictions[pos_mask], targets[pos_mask])
        if neg_mask.any():
            metrics['neg_accuracy'] = calculate_accuracy(predictions[neg_mask], targets[neg_mask])
        
        return dict(metrics)  # Convert defaultdict to regular dict
        
    except Exception as e:
        logging.error("Error calculating metrics:")
        logging.error(f"Predictions shape: {predictions.shape}")
        logging.error(f"Targets shape: {targets.shape}")
        logging.error(f"Error details: {str(e)}")
        raise

def calculate_hit_rate(preds: torch.Tensor, targets: torch.Tensor, k: int, batch_size: int) -> float:
    """Calculate Hit Rate @ K for batched predictions.
    
    A hit occurs when the positive item appears in the top-k predictions.
    """
    # Reshape into batches
    preds = preds.view(-1, batch_size)
    targets = targets.view(-1, batch_size)
    
    # Get top K indices
    _, top_indices = torch.topk(preds, k=min(k, preds.size(1)), dim=1)
    
    # Create mask of positive items
    target_mask = targets == 1
    
    # Check if positive items are in top K
    hits = torch.zeros(preds.size(0), dtype=torch.bool)
    for i in range(preds.size(0)):
        hits[i] = target_mask[i, top_indices[i]].any()
    
    return hits.float().mean().item()

def calculate_ndcg(preds: torch.Tensor, targets: torch.Tensor, k: int, batch_size: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain @ K.
    
    NDCG measures the quality of ranking considering position importance.
    """
    # Reshape into batches
    preds = preds.view(-1, batch_size)
    targets = targets.view(-1, batch_size)
    
    # Get top K indices
    _, indices = torch.topk(preds, k=min(k, preds.size(1)), dim=1)
    
    # Calculate DCG
    positions = torch.arange(1, indices.size(1) + 1, dtype=torch.float32)
    weights = 1 / torch.log2(positions + 1)
    
    dcg = torch.zeros(preds.size(0))
    for i in range(preds.size(0)):
        relevant_items = targets[i, indices[i]]
        dcg[i] = (relevant_items * weights).sum()
    
    # Calculate ideal DCG
    ideal_dcg = torch.zeros(preds.size(0))
    ideal_relevant = torch.ones(min(k, int(targets.sum(dim=1).max().item())))
    if len(ideal_relevant) > 0:
        ideal_dcg += (ideal_relevant * weights[:len(ideal_relevant)]).sum()
    
    # Calculate NDCG
    ndcg = dcg / ideal_dcg.clamp(min=1e-8)  # Avoid division by zero
    return ndcg.mean().item()

def calculate_mrr(preds: torch.Tensor, targets: torch.Tensor, k: int, batch_size: int) -> float:
    """Calculate Mean Reciprocal Rank @ K.
    
    MRR measures where the first relevant item appears in the ranking.
    """
    # Reshape into batches
    preds = preds.view(-1, batch_size)
    targets = targets.view(-1, batch_size)
    
    # Get top K indices
    _, indices = torch.topk(preds, k=min(k, preds.size(1)), dim=1)
    
    # Calculate reciprocal ranks
    rr = torch.zeros(preds.size(0))
    for i in range(preds.size(0)):
        # Find first positive item in top K
        target_positions = targets[i, indices[i]]
        first_pos = torch.nonzero(target_positions, as_tuple=True)[0]
        if len(first_pos) > 0:
            rr[i] = 1 / (first_pos[0].item() + 1)
    
    return rr.mean().item()

def calculate_map(preds: torch.Tensor, targets: torch.Tensor, k: int, batch_size: int) -> float:
    """Calculate Mean Average Precision @ K.
    
    MAP measures the average precision at each relevant item in the ranking.
    """
    # Reshape into batches
    preds = preds.view(-1, batch_size)
    targets = targets.view(-1, batch_size)
    
    # Get top K indices
    _, indices = torch.topk(preds, k=min(k, preds.size(1)), dim=1)
    
    # Calculate AP for each batch
    ap = torch.zeros(preds.size(0))
    for i in range(preds.size(0)):
        relevant_items = targets[i, indices[i]]
        if relevant_items.sum() == 0:
            continue
            
        # Calculate precision at each position
        precisions = torch.cumsum(relevant_items, dim=0) / torch.arange(1, len(relevant_items) + 1)
        ap[i] = (precisions * relevant_items).sum() / relevant_items.sum()
    
    return ap.mean().item()

def calculate_auc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Area Under the ROC Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(targets.numpy(), preds.numpy())
    except ImportError:
        logging.warning("scikit-learn not available, calculating AUC manually")
        
        # Manual AUC calculation
        pos_preds = preds[targets == 1]
        neg_preds = preds[targets == 0]
        
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return 0.5
            
        pos_preds = pos_preds.unsqueeze(1)
        neg_preds = neg_preds.unsqueeze(0)
        
        comparisons = (pos_preds > neg_preds).float()
        tie_correction = 0.5 * (pos_preds == neg_preds).float()
        
        return (comparisons.sum() + tie_correction.sum()) / (len(pos_preds) * len(neg_preds))

def calculate_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate binary classification accuracy."""
    return ((preds >= threshold) == targets).float().mean().item()