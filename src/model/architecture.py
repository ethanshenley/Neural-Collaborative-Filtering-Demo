import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, PoolingType
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
                dropout: float = 0.2,
                negative_samples: int = 4):  # Add negative_samples parameter
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
        self.negative_samples = negative_samples  # Store as instance attribute
        
        # MF path embeddings
        self.mf_embedding_collection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="user_id",
                    embedding_dim=mf_embedding_dim,
                    num_embeddings=num_users,
                    feature_names=["user_id"],
                    pooling=PoolingType.SUM
                ),
                EmbeddingBagConfig(
                    name="product_id",
                    embedding_dim=mf_embedding_dim,
                    num_embeddings=num_products,
                    feature_names=["product_id"],
                    pooling=PoolingType.SUM
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
                    feature_names=["user_id"],
                    pooling=PoolingType.SUM
                ),
                EmbeddingBagConfig(
                    name="product_id",
                    embedding_dim=mlp_embedding_dim,
                    num_embeddings=num_products,
                    feature_names=["product_id"],
                    pooling=PoolingType.SUM
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
        mlp_layers = []
        current_dim = combined_dim  # Start with combined input dimension

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
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
        """Forward pass with robust batch handling and shape validation.
        
        Args:
            features: KeyedJaggedTensor containing user_id and product_id features
                     Shape: (batch_size * (1 + negative_samples))
            
        Returns:
            Tensor of prediction scores
            Shape: (batch_size * (1 + negative_samples), 1)
            
        Raises:
            ValueError: If tensor shapes are inconsistent
        """
        try:
            # Calculate batch dimensions
            total_samples = features.values().size(0) // 2  # Divide by 2 because we have user_id and product_id
            samples_per_interaction = 1 + self.negative_samples if self.training else 1
            batch_size = total_samples // samples_per_interaction
            
            # Log shape information
            logging.debug(f"Forward pass dimensions:")
            logging.debug(f"- Total samples: {total_samples}")
            logging.debug(f"- Batch size: {batch_size}")
            logging.debug(f"- Samples per interaction: {samples_per_interaction}")
            
            # Get embeddings with validation
            try:
                mf_embeddings = self.mf_embedding_collection(features)
                mlp_embeddings = self.mlp_embedding_collection(features)
            except Exception as e:
                logging.error("Failed to get embeddings:")
                logging.error(f"Feature values shape: {features.values().shape}")
                logging.error(f"Feature lengths: {features.lengths()}")
                raise
            
            # Validate embedding shapes
            for path, embeddings in [("MF", mf_embeddings), ("MLP", mlp_embeddings)]:
                for key in embeddings.keys():
                    tensor = embeddings[key]
                    if tensor.size(0) != total_samples:
                        raise ValueError(
                            f"{path} {key} embedding shape mismatch: "
                            f"got {tensor.size(0)}, expected {total_samples}"
                        )
            
            # Process MF path
            user_mf = self.mf_norm(mf_embeddings["user_id"])  # [batch_size * samples_per_interaction, embed_dim]
            product_mf = self.mf_norm(mf_embeddings["product_id"])
            mf_vector = user_mf * product_mf  # Element-wise multiplication
            mf_pred = self.mf_output(mf_vector)  # [batch_size * samples_per_interaction, 1]
            
            # Process MLP path with attention
            user_mlp = self.mlp_norm(mlp_embeddings["user_id"])  
            product_mlp = self.mlp_norm(mlp_embeddings["product_id"])
            
            # Reshape for attention
            user_mlp = user_mlp.view(batch_size, samples_per_interaction, -1)
            product_mlp = product_mlp.view(batch_size, samples_per_interaction, -1)
            
            # Apply attention across samples for each batch
            user_product_attn = self.user_product_attention(
                user_mlp,  # [batch_size, samples_per_interaction, embed_dim]
                product_mlp,
                product_mlp
            )  # [batch_size, samples_per_interaction, embed_dim]
            
            # Reshape attention output
            user_product_attn = user_product_attn.view(total_samples, -1)
            
            # Create temporal features
            temporal_embeds = torch.zeros(
                total_samples, 
                self.temporal_dim,
                device=user_mf.device,
                dtype=user_mf.dtype
            )
            
            # Combine features for MLP
            combined_features = torch.cat([
                user_product_attn,
                temporal_embeds
            ], dim=1)
            
            # Process through MLP layers
            try:
                mlp_vector = self.mlp(combined_features)  # [total_samples, mlp_hidden_dims[-1]]
                mlp_pred = self.mlp_output(mlp_vector)    # [total_samples, 1]
            except Exception as e:
                logging.error("Error in MLP processing:")
                logging.error(f"Combined features shape: {combined_features.shape}")
                logging.error(f"MLP hidden dims: {self.mlp_hidden_dims}")
                raise
            
            # Combine predictions
            combined = torch.cat([mf_pred, mlp_pred], dim=1)  # [total_samples, 2]
            outputs = self.final(combined)  # [total_samples, 1]
            
            # Final validation
            expected_shape = (total_samples, 1)
            if outputs.shape != expected_shape:
                raise ValueError(
                    f"Output shape mismatch: got {outputs.shape}, "
                    f"expected {expected_shape}"
                )
                
            # Add debugging info for first forward pass
            if not hasattr(self, '_first_forward_done'):
                logging.info(f"First forward pass shapes:")
                logging.info(f"- User MF: {user_mf.shape}")
                logging.info(f"- Product MF: {product_mf.shape}")
                logging.info(f"- User-Product Attention: {user_product_attn.shape}")
                logging.info(f"- MLP Vector: {mlp_vector.shape}")
                logging.info(f"- Final Output: {outputs.shape}")
                self._first_forward_done = True
            
            return outputs
            
        except Exception as e:
            logging.error("Error in model forward pass:")
            logging.error(f"Input features: {features}")
            logging.error(f"Training mode: {self.training}")
            logging.error(f"Error details: {str(e)}")
            raise
            
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

    def forward_simple(self, user_ids, product_ids):
        """Simple forward pass with direct tensor inputs"""
        user_emb = self.mf_user_embedding(user_ids)
        product_emb = self.mf_product_embedding(product_ids)
        
        # Compute dot product
        dot_product = (user_emb * product_emb).sum(dim=1)
        return dot_product