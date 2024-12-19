# Introduction

Our `AdvancedNCF` model implements a sophisticated hybrid architecture that combines the strengths of traditional matrix factorization with deep learning components, specifically designed for the convenience store recommendation context.

## Dual Embedding Paths

``` python
# Base embeddings (MF path)
self.mf_embedding_collection = EmbeddingBagCollection([
    EmbeddingBagConfig(name="user_mf_embeddings", ...),
    EmbeddingBagConfig(name="product_mf_embeddings", ...)
])

# Neural path embeddings
self.mlp_embedding_collection = EmbeddingBagCollection([
    EmbeddingBagConfig(name="user_mlp_embeddings", ...),
    EmbeddingBagConfig(name="product_mlp_embeddings", ...)
])
```
The model maintains separate embedding spaces for matrix factorization (MF) and multi-layer perceptron (MLP) paths. This separation is necessary because:

- MF path captures linear relationships through element-wise products
- MLP path learns non-linear feature interactions
- Independent embeddings allow each path to specialize in different aspects of user-item relationships

## Temporal Encoding
```python
class TemporalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_period: int = 365):
        # Seasonal patterns
        position = torch.arange(max_period).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_period, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
```
The temporal encoding mechanism:

- Uses sinusoidal position encoding to capture cyclical patterns
- Handles multiple time granularities (hour, day, month)
- Enables the model to learn both short-term and seasonal patterns
- Maintains constant embedding dimensionality regardless of time span

## Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
```

The attention mechanism serves multiple purposes:

- Captures dynamic user-product interactions
- Learns importance weights for different aspects of user behavior
- Enables the model to focus on relevant parts of purchase history
- Multiple heads allow attention to different feature subspaces

## Category Hierarchy
```python
class CategoryHierarchy(nn.Module):
    def __init__(self, num_departments: int, num_categories: int, embed_dim: int):
        self.department_embed = nn.Embedding(num_departments, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        self.hierarchy_attn = MultiHeadAttention(embed_dim)
```
The hierarchical representation:

- Models relationships between departments and categories
- Enables cross-category learning
- Captures product similarities at different granularities
- Uses attention to dynamically weight hierarchical relationships

## Feature Combination and MLP
```python
combined_dim = (
    mf_embedding_dim +      # MF path
    mlp_embedding_dim +     # MLP path
    mlp_embedding_dim +     # Category hierarchy
    temporal_dim           # Temporal features
)

self.feature_combination = nn.Sequential(
    nn.Linear(combined_dim, mlp_hidden_dims[0]),
    nn.ReLU(),
    nn.BatchNorm1d(mlp_hidden_dims[0]),
    nn.Dropout(dropout)
)
```
The combination network:

- Integrates signals from all feature types
- Uses batch normalization for stable training
- Applies dropout for regularization
- Gradually reduces dimensionality through the network

## Dual Output Paths
```python
# Output paths
self.mf_output = nn.Linear(mf_embedding_dim, 1)
self.mlp_output = nn.Linear(mlp_hidden_dims[-1], 1)

# Final combination
self.final = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
```

The dual output structure:

- Allows MF and MLP paths to contribute independently
- Learns optimal weighting between paths
- Uses sigmoid activation for final probability scores

## Training and Inference Flow
1. User and product features are processed through respective embedding paths
2. Temporal features are encoded and combined with embeddings
3. Attention mechanisms process sequential and hierarchical relationships
4. MF path computes direct user-item compatibility
5. MLP path learns complex feature interactions
6. Final layer combines both paths for recommendation scores

## Advantages
- Handles both static and dynamic features effectively
- Captures both linear and non-linear relationships
- Scales efficiently with growing product catalogs
- Balances memorization (MF) and generalization (MLP)
- Supports both batch training and real-time inference