# src/utils/metrics.py

import torch
import numpy as np
from typing import Dict, List

def calculate_hit_rate(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Hit Rate @ K
    Args:
        predictions: Predicted scores (B, N)
        targets: Binary target values (B, N)
        k: Number of items to consider
    Returns:
        Hit Rate @ K
    """
    # Get top K item indices
    _, top_indices = torch.topk(predictions, k, dim=1)
    
    # Check if any target item is in top K
    target_indices = torch.nonzero(targets, as_tuple=True)[1]
    hits = [(target in top_k) for target, top_k in zip(target_indices, top_indices)]
    
    return torch.tensor(hits).float().mean().item()

def calculate_ndcg(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain @ K
    Args:
        predictions: Predicted scores (B, N)
        targets: Binary target values (B, N)
        k: Number of items to consider
    Returns:
        NDCG @ K
    """
    # Get top K item indices
    _, top_indices = torch.topk(predictions, k, dim=1)
    
    # Calculate DCG
    dcg = torch.zeros(len(predictions))
    for i, (top_k, target) in enumerate(zip(top_indices, targets)):
        relevance = target[top_k]
        position_discount = 1 / torch.log2(torch.arange(len(top_k), device=predictions.device) + 2)
        dcg[i] = (relevance * position_discount).sum()
    
    # Calculate ideal DCG
    ideal_dcg = torch.zeros(len(predictions))
    target_indices = torch.nonzero(targets, as_tuple=True)[1]
    position_discount = 1 / torch.log2(torch.arange(1, k + 1, device=predictions.device) + 1)
    ideal_dcg = position_discount.sum()
    
    return (dcg / ideal_dcg).mean().item()

def calculate_mrr(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank @ K
    Args:
        predictions: Predicted scores (B, N)
        targets: Binary target values (B, N)
        k: Number of items to consider
    Returns:
        MRR @ K
    """
    # Get top K item indices
    _, top_indices = torch.topk(predictions, k, dim=1)
    
    # Find rank of target item
    target_indices = torch.nonzero(targets, as_tuple=True)[1]
    reciprocal_ranks = []
    
    for target, top_k in zip(target_indices, top_indices):
        try:
            rank = (top_k == target).nonzero()[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        except:
            reciprocal_ranks.append(0.0)
            
    return torch.tensor(reciprocal_ranks).mean().item()

def calculate_map(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Calculate Mean Average Precision @ K
    Args:
        predictions: Predicted scores (B, N)
        targets: Binary target values (B, N)
        k: Number of items to consider
    Returns:
        MAP @ K
    """
    # Get top K item indices
    _, top_indices = torch.topk(predictions, k, dim=1)
    
    # Calculate AP for each prediction
    aps = []
    for top_k, target in zip(top_indices, targets):
        relevance = target[top_k]
        if relevance.sum() == 0:
            aps.append(0.0)
            continue
            
        precision_at_k = torch.cumsum(relevance, dim=0) / torch.arange(1, k + 1, device=predictions.device)
        ap = (precision_at_k * relevance).sum() / relevance.sum()
        aps.append(ap.item())
        
    return torch.tensor(aps).mean().item()

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> Dict[str, float]:
    """Calculate all recommendation metrics"""
    return {
        'hit_rate': calculate_hit_rate(predictions, targets, k),
        'ndcg': calculate_ndcg(predictions, targets, k),
        'mrr': calculate_mrr(predictions, targets, k),
        'map': calculate_map(predictions, targets, k),
    }

def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics"""
    print("\nEvaluation Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:>20}: {value:.4f}")
    print("-" * 40)