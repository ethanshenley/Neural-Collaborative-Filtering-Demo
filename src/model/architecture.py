import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
except ImportError:
    # Fallback to basic PyTorch for single-GPU training
    from torch.nn import EmbeddingBag as EmbeddingBagCollection
    EmbeddingBagConfig = dict
    
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict, List, Optional, Tuple
import math

import logging

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim) # query proj
        self.k_proj = nn.Linear(embed_dim, embed_dim) # key proj
        self.v_proj = nn.Linear(embed_dim, embed_dim) # value proj
        self.out_proj = nn.Linear(embed_dim, embed_dim) # output proj
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)

class TemporalEncoding(nn.Module):
    '''
    This class encodes temporal features, specifically time & season
    Time features are embedded into dimensions matching their length
    e.g hours = 24, days = 7, month = 12
    Seasonal patterns are embedded similarly, with 365 days and 4 seasons
    squeezed.
    
    '''
    def __init__(self, embed_dim: int, max_period: int = 365):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Time embeddings
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.day_embed = nn.Embedding(7, embed_dim)
        self.month_embed = nn.Embedding(12, embed_dim)
        
        # Seasonal patterns
        position = torch.arange(max_period).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_period, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, hour: torch.Tensor, day: torch.Tensor, 
                month: torch.Tensor, days_since: torch.Tensor) -> torch.Tensor:
        temporal = (self.hour_embed(hour) + 
                   self.day_embed(day) + 
                   self.month_embed(month))
        
        # Add seasonal patterns
        seasonal = self.pe[days_since.long() % self.max_period]
        return temporal + seasonal

class CategoryHierarchy(nn.Module):
    def __init__(self, 
                 num_departments: int,
                 num_categories: int,
                 embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.department_embed = nn.Embedding(num_departments, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        
        self.hierarchy_attn = MultiHeadAttention(embed_dim, num_heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, department_ids: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        dept_embeds = self.department_embed(department_ids)
        cat_embeds = self.category_embed(category_ids)
        
        # Combine using attention
        hierarchy_embeds = self.hierarchy_attn(cat_embeds, dept_embeds, dept_embeds)
        hierarchy_embeds = self.dropout(hierarchy_embeds)
        
        return self.norm(hierarchy_embeds + cat_embeds)  # Residual connection

class AdvancedNCF(nn.Module):
    def __init__(self, 
                num_users: int,
                num_products: int,
                num_departments: int,
                num_categories: int,
                mf_embedding_dim: int = 64,
                mlp_embedding_dim: int = 64,
                temporal_dim: int = 32,
                mlp_hidden_dims: List[int] = [256, 128, 64],
                num_heads: int = 4,
                dropout: float = 0.2):
        super().__init__()
        
        # Save all configuration parameters
        self.num_users = num_users
        self.num_products = num_products
        self.num_departments = num_departments
        self.num_categories = num_categories
        self.mf_embedding_dim = mf_embedding_dim
        self.mlp_embedding_dim = mlp_embedding_dim
        # Layer normalizations for embeddings
        self.mf_norm = nn.LayerNorm(mf_embedding_dim)
        self.mlp_norm = nn.LayerNorm(mlp_embedding_dim)
        self.temporal_dim = temporal_dim
        self.mlp_hidden_dims = mlp_hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # MF path embeddings
        self.mf_embedding_collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="user_id",
                    embedding_dim=mf_embedding_dim,
                    num_embeddings=num_users,
                    feature_names=["user_id"]
                ),
                EmbeddingBagConfig(
                    name="product_id",
                    embedding_dim=mf_embedding_dim,
                    num_embeddings=num_products,
                    feature_names=["product_id"]
                )
            ]
        )
        
        # MLP path embeddings
        self.mlp_embedding_collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="user_id",
                    embedding_dim=mlp_embedding_dim,
                    num_embeddings=num_users,
                    feature_names=["user_id"]
                ),
                EmbeddingBagConfig(
                    name="product_id",
                    embedding_dim=mlp_embedding_dim,
                    num_embeddings=num_products,
                    feature_names=["product_id"]
                )
            ]
        )
        
        # Category hierarchy
        self.category_hierarchy = CategoryHierarchy(
            num_departments=num_departments,
            num_categories=num_categories,
            embed_dim=mlp_embedding_dim,
            dropout=dropout
        )
        
        # Temporal features
        self.temporal_encoding = TemporalEncoding(temporal_dim)
        
        # Attention mechanisms
        self.user_product_attention = MultiHeadAttention(
            embed_dim=mlp_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.sequence_attention = MultiHeadAttention(
            embed_dim=mlp_embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Initialize neural network layers with proper shapes
        combined_dim = (
            mlp_embedding_dim +    # From user-product attention
            temporal_dim          # From temporal features
        )
        
        self.feature_combination = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_dims[0]),  # Changed from BatchNorm1d
            nn.Dropout(dropout)
        )     

        # Main MLP
        layers = []
        input_dim = mlp_hidden_dims[0]

        for hidden_dim in mlp_hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # Output layers
        self.mf_output = nn.Linear(mf_embedding_dim, 1)
        self.mlp_output = nn.Linear(mlp_hidden_dims[-1], 1)
        
        # Final combination
        self.final = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalizations
        self.mf_norm = nn.LayerNorm(mf_embedding_dim)
        self.mlp_norm = nn.LayerNorm(mlp_embedding_dim)
        
    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        """Forward pass of the NCF model.
        
        Args:
            features: KeyedJaggedTensor containing user and product ids
            
        Returns:
            Tensor of shape [B, 1] containing prediction scores
        """
        # Get embeddings from collections
        mf_embeddings = self.mf_embedding_collection(features)
        mlp_embeddings = self.mlp_embedding_collection(features)
        
        # Get batch size from embeddings correctly
        batch_size = mf_embeddings["user_id"].size(0)
        
        # Add batch size logging/checking
        if batch_size < 2:
            logging.warning(f"Small batch detected: {batch_size}")
        
        # Get embeddings and apply layer norm
        user_mf = self.mf_norm(mf_embeddings["user_id"])  
        product_mf = self.mf_norm(mf_embeddings["product_id"])
        user_mlp = self.mlp_norm(mlp_embeddings["user_id"])
        product_mlp = self.mlp_norm(mlp_embeddings["product_id"])

        # Matrix Factorization path
        mf_vector = user_mf * product_mf  # [B, embed_dim]
        mf_pred = self.mf_output(mf_vector)  # [B, 1]
        
        # Neural Network path with attention
        user_mlp = user_mlp.unsqueeze(1)  # [B, 1, embed_dim]
        product_mlp = product_mlp.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Apply attention
        user_product_attn = self.user_product_attention(
            user_mlp, product_mlp, product_mlp
        ).squeeze(1)  # [B, embed_dim]
        
        # Create temporal embeddings
        temporal_embeds = torch.zeros(
            batch_size, self.temporal_dim, 
            device=user_mf.device
        )  # [B, temporal_dim]
        
        # Combine features
        combined_features = torch.cat([
            user_product_attn,  # [B, embed_dim]
            temporal_embeds,    # [B, temporal_dim]
        ], dim=1)  # [B, embed_dim + temporal_dim]
        
        # Pass through MLP
        combined = self.feature_combination(combined_features)  # [B, hidden_dim]
        mlp_vector = self.mlp(combined)  # [B, hidden_dim]
        mlp_pred = self.mlp_output(mlp_vector)  # [B, 1]
        
        # Final prediction
        final_pred = torch.cat([mf_pred, mlp_pred], dim=1)  # [B, 2]
        return self.final(final_pred)  # [B, 1]
        
    def get_user_embeddings(self, user_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get user embeddings for efficient inference"""
        mf_embeddings = self.mf_embedding_collection(user_features["user_features"])
        mlp_embeddings = self.mlp_embedding_collection(user_features["user_features"])
        
        return {
            "mf": self.mf_norm(mf_embeddings["user_id"]),
            "mlp": self.mlp_norm(mlp_embeddings["user_id"])
        }
    
    def get_product_embeddings(self, product_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get product embeddings for efficient inference"""
        mf_embeddings = self.mf_embedding_collection(product_features["product_features"])
        mlp_embeddings = self.mlp_embedding_collection(product_features["product_features"])
        
        category_embeds = self.category_hierarchy(
            product_features["category_features"]["department_ids"],
            product_features["category_features"]["category_ids"]
        )
        
        return {
            "mf": self.mf_norm(mf_embeddings["product_id"]),
            "mlp": self.mlp_norm(mlp_embeddings["product_id"]),
            "category": category_embeds
        }