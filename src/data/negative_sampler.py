import torch
import numpy as np
from typing import Tuple, List
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class NegativeSampler:
    def __init__(self, num_products: int, num_negative: int = 4):
        self.num_products = num_products
        self.num_negative = num_negative
    
    def sample(self, 
               positive_features: KeyedJaggedTensor, 
               positive_items: torch.Tensor) -> Tuple[KeyedJaggedTensor, torch.Tensor]:
        """Generate negative samples for each positive interaction."""
        batch_size = len(positive_items)
        
        # Generate random negative items
        neg_items = torch.randint(
            0, self.num_products, 
            (batch_size, self.num_negative)
        )
        
        # Create features for negative samples
        user_ids = positive_features.values[::2].repeat(self.num_negative)
        product_ids = neg_items.flatten()
        
        values = torch.stack([user_ids, product_ids], dim=1).flatten()
        lengths = torch.tensor([1, 1], dtype=torch.int32).repeat(batch_size * self.num_negative)
        
        neg_features = KeyedJaggedTensor(
            keys=["user_id", "product_id"],
            values=values,
            lengths=lengths
        )
        
        # Create targets (0 for negative samples)
        neg_targets = torch.zeros(batch_size * self.num_negative)
        
        return neg_features, neg_targets