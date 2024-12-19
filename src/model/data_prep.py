# src/data/data_prep.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

class SheetzDataset(Dataset):
    def __init__(
        self,
        interactions_df,
        user_stats_df,
        product_stats_df,
        mode: str = 'train',
        validation_days: int = 10,
        negative_samples: int = 4
    ):
        """
        Args:
            interactions_df: DataFrame from user_interactions view
            user_stats_df: DataFrame from user_statistics view
            product_stats_df: DataFrame from product_statistics view
            mode: One of ['train', 'val', 'test']
            validation_days: Number of days to use for validation/test
            negative_samples: Number of negative samples per positive sample
        """
        self.mode = mode
        self.negative_samples = negative_samples
        
        # Split data based on time
        latest_date = interactions_df['transaction_timestamp'].max()
        split_date = latest_date - timedelta(days=validation_days)
        
        if mode == 'train':
            self.interactions = interactions_df[
                interactions_df['transaction_timestamp'] < split_date
            ]
        else:
            self.interactions = interactions_df[
                interactions_df['transaction_timestamp'] >= split_date
            ]
            
        # Create user and product mappings
        self.user_to_idx = {
            user: idx for idx, user in enumerate(user_stats_df['user_id'].unique())
        }
        self.product_to_idx = {
            prod: idx for idx, prod in enumerate(product_stats_df['product_id'].unique())
        }
        
        # Create product popularity scores for negative sampling
        total_sales = product_stats_df['total_sales'].sum()
        self.product_weights = (
            (1 / product_stats_df['total_sales']) / 
            (1 / total_sales)
        ).values
        
        # Store dimensions
        self.num_users = len(self.user_to_idx)
        self.num_products = len(self.product_to_idx)
        
        # Create interaction list
        self.interaction_list = self._create_interaction_list()
        
    def _create_interaction_list(self) -> List[Tuple[int, int, float]]:
        """Create list of (user_idx, product_idx, amount) tuples"""
        interactions = []
        
        for _, row in self.interactions.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            product_idx = self.product_to_idx[row['product_id']]
            amount = row['amount']
            interactions.append((user_idx, product_idx, amount))
            
        return interactions
        
    def _sample_negative(self, user_idx: int, positive_product: int) -> int:
        """Sample a negative product for a user"""
        while True:
            # Sample based on inverse popularity
            product_idx = np.random.choice(
                len(self.product_to_idx),
                p=self.product_weights
            )
            # Check if this is a positive interaction
            if product_idx != positive_product:
                return product_idx
    
    def __len__(self) -> int:
        return len(self.interaction_list)
    
    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, torch.Tensor]:
        user_idx, product_idx, amount = self.interaction_list[idx]
        
        # For training, generate negative samples
        if self.mode == 'train':
            # Create positive sample
            user_indices = [user_idx]
            product_indices = [product_idx]
            targets = [1.0]
            
            # Generate negative samples
            for _ in range(self.negative_samples):
                neg_product = self._sample_negative(user_idx, product_idx)
                user_indices.append(user_idx)
                product_indices.append(neg_product)
                targets.append(0.0)
                
            # Create KeyedJaggedTensor
            features = KeyedJaggedTensor(
                keys=["user_id", "product_id"],
                values=torch.tensor(user_indices + product_indices),
                lengths=torch.tensor([1] * len(user_indices) * 2)  # One per feature
            )
            
            return features, torch.tensor(targets)
        
        else:
            # For validation/testing, just return the positive sample
            features = KeyedJaggedTensor(
                keys=["user_id", "product_id"],
                values=torch.tensor([user_idx, product_idx]),
                lengths=torch.tensor([1, 1])  # One per feature
            )
            
            return features, torch.tensor([1.0])

def create_data_loaders(
    interactions_df,
    user_stats_df,
    product_stats_df,
    batch_size: int = 256,
    num_workers: int = 4,
    validation_days: int = 10,
    negative_samples: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    train_dataset = SheetzDataset(
        interactions_df=interactions_df,
        user_stats_df=user_stats_df,
        product_stats_df=product_stats_df,
        mode='train',
        validation_days=validation_days,
        negative_samples=negative_samples
    )
    
    val_dataset = SheetzDataset(
        interactions_df=interactions_df,
        user_stats_df=user_stats_df,
        product_stats_df=product_stats_df,
        mode='val',
        validation_days=validation_days
    )
    
    test_dataset = SheetzDataset(
        interactions_df=interactions_df,
        user_stats_df=user_stats_df,
        product_stats_df=product_stats_df,
        mode='test',
        validation_days=validation_days
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader