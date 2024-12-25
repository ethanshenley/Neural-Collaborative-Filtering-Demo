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
    batch_size: Optional[int] = None,
    negative_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate various recommendation metrics (Hit@K, NDCG@K, MRR@K, MAP@K)
    plus AUC and overall accuracy.

    Args:
        predictions: Model scores, shape (batch_size*M,) or (batch_size*M, 1)
        targets: Ground truth, same shape as predictions
        k_values: List of top-k values
        batch_size: How many users in this batch
        negative_samples: How many negative items per user
            => total items per user = 1 + negative_samples

    Returns:
        Dictionary of metrics.
    """
    try:
        # Move everything to CPU, remove extra dimension if needed
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        if predictions.dim() == 2 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # In a typical scenario, total_samples = batch_size * (1 + negative_samples)
        if batch_size is None or negative_samples is None:
            raise ValueError(
                "Please provide both batch_size and negative_samples "
                "to reshape predictions into [batch_size, 1+negative_samples]."
            )

        M = 1 + negative_samples
        expected_size = batch_size * M
        if predictions.numel() != expected_size:
            raise ValueError(
                f"Size mismatch: got {predictions.numel()} total preds, "
                f"but expected batch_size*M = {expected_size}."
            )

        # Reshape predictions and targets into [batch_size, M]
        predictions = predictions.view(batch_size, M)
        targets = targets.view(batch_size, M)

        # Prepare a dict for storing results
        metric_dict = {}

        # For each K in k_values, compute top-K style metrics
        for k in k_values:
            # Hit Rate @ K
            hit_rate = calculate_hit_rate(predictions, targets, k)
            metric_dict[f"hit_rate@{k}"] = hit_rate

            # NDCG @ K
            ndcg_k = calculate_ndcg(predictions, targets, k)
            metric_dict[f"ndcg@{k}"] = ndcg_k

            # MRR @ K
            mrr_k = calculate_mrr(predictions, targets, k)
            metric_dict[f"mrr@{k}"] = mrr_k

            # MAP @ K
            map_k = calculate_map(predictions, targets, k)
            metric_dict[f"map@{k}"] = map_k

        # Classification metrics: AUC, overall accuracy, pos/neg accuracy
        # Flatten again to (N,) for these classification metrics:
        flat_preds = predictions.view(-1)
        flat_targs = targets.view(-1)

        metric_dict["auc"] = calculate_auc(flat_preds, flat_targs)
        metric_dict["accuracy"] = calculate_accuracy(flat_preds, flat_targs)

        # If you want pos/neg accuracy:
        pos_mask = flat_targs == 1
        neg_mask = flat_targs == 0
        if pos_mask.any():
            metric_dict["pos_accuracy"] = calculate_accuracy(
                flat_preds[pos_mask], flat_targs[pos_mask]
            )
        if neg_mask.any():
            metric_dict["neg_accuracy"] = calculate_accuracy(
                flat_preds[neg_mask], flat_targs[neg_mask]
            )

        return metric_dict

    except Exception as e:
        logging.error("Error calculating metrics:")
        logging.error(f"Predictions shape: {predictions.shape if 'predictions' in locals() else 'unknown'}")
        logging.error(f"Targets shape: {targets.shape if 'targets' in locals() else 'unknown'}")
        logging.error(f"Error details: {str(e)}")
        raise

def calculate_hit_rate(
    preds_2d: torch.Tensor,
    targs_2d: torch.Tensor,
    k: int
) -> float:
    """
    preds_2d, targs_2d shape: [batch_size, M]
    We pick the top k items for each row (user).
    Then check if there's a '1' in those top k items.
    Return average over batch_size.
    """
    batch_size, M = preds_2d.shape
    k = min(k, M)

    # Get topk indices for each user
    _, top_indices = torch.topk(preds_2d, k=k, dim=1)  # shape => [batch_size, k]

    # Check if any of these indices is a positive
    hits = 0
    for i in range(batch_size):
        # user i's top-k item indices
        user_topk_idx = top_indices[i]    # shape => [k]
        # see if any is 1
        if (targs_2d[i, user_topk_idx] == 1).any():
            hits += 1

    return hits / batch_size

def calculate_ndcg(
    preds_2d: torch.Tensor,
    targs_2d: torch.Tensor,
    k: int
) -> float:
    """
    NDCG @ K, row = user, col = items.
    """
    batch_size, M = preds_2d.shape
    k = min(k, M)

    # Sort by predicted score, descending
    _, sorted_indices = torch.sort(preds_2d, dim=1, descending=True)

    # For each user, compute DCG of top-k
    ndcg_list = []
    for i in range(batch_size):
        # sorted_indices[i] is array of item indices from highest pred to lowest
        top_k_idx = sorted_indices[i, :k]
        # relevant items
        relevant = targs_2d[i, top_k_idx]
        # positions for DCG
        pos = torch.arange(1, k+1, dtype=torch.float32)
        # DCG = sum( rel_i / log2(1 + position) )
        # Here, rel_i is {0 or 1}
        # positions => [1..k]
        dcg = (relevant / torch.log2(pos + 1)).sum().item()

        # ideal DCG => sort the user's items by true relevance descending
        # But if exactly 1 positive per user, ideal DCG for top-k is 1/log2(1+pos_of_that_positive)
        # For multiple positives, we do:
        ideal_sorted = torch.sort(targs_2d[i], descending=True).values
        ideal_topk = ideal_sorted[:k]
        idcg = (ideal_topk / torch.log2(pos + 1)).sum().item()
        if idcg <= 0:
            ndcg_list.append(0.0)
        else:
            ndcg_list.append(dcg / idcg)

    return float(np.mean(ndcg_list))

def calculate_mrr(
    preds_2d: torch.Tensor,
    targs_2d: torch.Tensor,
    k: int
) -> float:
    """
    Mean Reciprocal Rank @ K.
    """
    batch_size, M = preds_2d.shape
    k = min(k, M)

    _, sorted_indices = torch.sort(preds_2d, dim=1, descending=True)

    rr_list = []
    for i in range(batch_size):
        user_topk_idx = sorted_indices[i, :k]
        # targs_2d[i, user_topk_idx] is e.g. [1, 0, 0, ...]
        # find first positive
        positives = (targs_2d[i, user_topk_idx] == 1).nonzero(as_tuple=True)[0]
        if len(positives) > 0:
            # The first occurrence
            first_rank = positives[0].item()  # 0-based
            rr_list.append(1.0 / (first_rank + 1.0))
        else:
            rr_list.append(0.0)

    return float(np.mean(rr_list))

def calculate_map(
    preds_2d: torch.Tensor,
    targs_2d: torch.Tensor,
    k: int
) -> float:
    """
    Mean Average Precision @ K.
    """
    batch_size, M = preds_2d.shape
    k = min(k, M)

    _, sorted_indices = torch.sort(preds_2d, dim=1, descending=True)

    ap_list = []
    for i in range(batch_size):
        top_k_idx = sorted_indices[i, :k]
        relevant = targs_2d[i, top_k_idx]  # shape => [k], each is 0 or 1

        num_relevant = relevant.sum().item()
        if num_relevant == 0:
            ap_list.append(0.0)
            continue

        # precision at each position j
        # precision_j = (# relevant in top j) / j
        # average precision is mean of precision_j over each j where item is relevant
        running_sum = 0.0
        running_count = 0.0
        cume_relevant = 0
        for j in range(k):
            if relevant[j] == 1:
                cume_relevant += 1
                prec_j = cume_relevant / (j+1)
                running_sum += prec_j
                running_count += 1
        ap_list.append(running_sum / max(running_count, 1))

    return float(np.mean(ap_list))

def calculate_auc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Use sklearn if available, else a fallback approach.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(targets.numpy(), preds.numpy())
    except ImportError:
        logging.warning("scikit-learn not available, calculating AUC manually.")
        pos_preds = preds[targets == 1]
        neg_preds = preds[targets == 0]
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return 0.5
        pos_preds = pos_preds.unsqueeze(1)
        neg_preds = neg_preds.unsqueeze(0)
        # Comparisons
        comparisons = (pos_preds > neg_preds).float()
        ties = 0.5 * (pos_preds == neg_preds).float()
        auc_val = (comparisons.sum() + ties.sum()) / (pos_preds.numel() * neg_preds.numel())
        return auc_val.item()

def calculate_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Simple classification accuracy using a threshold.
    """
    return ((preds >= threshold) == targets).float().mean().item()