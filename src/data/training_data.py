# src/data/training_data.py

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class SheetzTrainingData(Dataset):
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        product_features_df: pd.DataFrame,
        mode: str = 'train',
        validation_days: int = 10,
        max_sequence_length: int = 50,
        negative_samples: int = 4
    ):
        self.mode = mode
        self.max_sequence_length = max_sequence_length
        self.negative_samples = negative_samples
        
        # Split data based on time
        latest_date = interactions_df['transaction_timestamp'].max()
        split_date = pd.Timestamp(latest_date) - pd.Timedelta(days=validation_days)
        
        if mode == 'train':
            self.interactions = interactions_df[
                interactions_df['transaction_timestamp'] < split_date
            ]
        else:
            self.interactions = interactions_df[
                interactions_df['transaction_timestamp'] >= split_date
            ]
            
        # Create feature mappings
        self.user_features = user_features_df
        self.product_features = product_features_df
        
        # Create indices
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_features_df['user_id'].unique())}
        self.product_to_idx = {pid: idx for idx, pid in enumerate(product_features_df['product_id'].unique())}
        
        # Create interaction list
        self.interaction_list = self._create_interaction_list()
        
    def _create_interaction_list(self) -> List[Dict]:
        interactions = []
        
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            timestamp = pd.Timestamp(row['transaction_timestamp'])
            
            # Get user features
            user_data = self.user_features[self.user_features['user_id'] == user_id].iloc[0]
            
            # Get product features
            product_data = self.product_features[self.product_features['product_id'] == product_id].iloc[0]
            
            # Get temporal features
            temporal_features = {
                'hour': timestamp.hour,
                'day': timestamp.dayofweek,
                'month': timestamp.month,
                'days_since': (pd.Timestamp(timestamp) - pd.Timestamp(user_data['first_interaction'])).days
            }
            
            # Get sequence features
            recent_interactions = eval(user_data['recent_interactions'])
            sequence_products = [
                self.product_to_idx[p['product_id']]
                for p in recent_interactions[:self.max_sequence_length]
                if p['transaction_timestamp'] < timestamp
            ]
            
            # Pad sequence if needed
            if len(sequence_products) < self.max_sequence_length:
                sequence_products.extend([0] * (self.max_sequence_length - len(sequence_products)))
            
            interactions.append({
                'user_idx': self.user_to_idx[user_id],
                'product_idx': self.product_to_idx[product_id],
                'temporal_features': temporal_features,
                'sequence_products': sequence_products,
                'department_idx': product_data['department_id'],
                'category_idx': product_data['category_id'],
                'amount': row['amount']
            })
            
        return interactions
        
    def _get_negative_samples(self, user_idx: int, positive_product: int) -> List[int]:
        """Get negative samples for a user"""
        negatives = []
        user_products = set(
            self.interactions[
                self.interactions['user_id'].map(self.user_to_idx) == user_idx
            ]['product_id'].map(self.product_to_idx)
        )
        
        while len(negatives) < self.negative_samples:
            neg = np.random.randint(0, len(self.product_to_idx))
            if neg not in user_products and neg != positive_product:
                negatives.append(neg)
                
        return negatives
    
    def __len__(self) -> int:
        return len(self.interaction_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        interaction = self.interaction_list[idx]
        
        # For training, add negative samples
        if self.mode == 'train':
            negative_products = self._get_negative_samples(
                interaction['user_idx'],
                interaction['product_idx']
            )
            
            # Create features for all products (positive + negatives)
            product_indices = [interaction['product_idx']] + negative_products
            targets = [1.0] + [0.0] * self.negative_samples
            
            # Repeat other features for each product
            return {
                'user_features': {
                    'user_id': torch.tensor([interaction['user_idx']] * (self.negative_samples + 1))
                },
                'product_features': {
                    'product_id': torch.tensor(product_indices)
                },
                'temporal_features': {
                    'hour': torch.tensor([interaction['temporal_features']['hour']] * (self.negative_samples + 1)),
                    'day': torch.tensor([interaction['temporal_features']['day']] * (self.negative_samples + 1)),
                    'month': torch.tensor([interaction['temporal_features']['month']] * (self.negative_samples + 1)),
                    'days_since': torch.tensor([interaction['temporal_features']['days_since']] * (self.negative_samples + 1))
                },
                'category_features': {
                    'department_ids': torch.tensor([interaction['department_idx']] * (self.negative_samples + 1)),
                    'category_ids': torch.tensor([interaction['category_idx']] * (self.negative_samples + 1))
                },
                'sequence_features': torch.tensor([interaction['sequence_products']] * (self.negative_samples + 1)),
                'targets': torch.tensor(targets)
            }
        else:
            # For validation/testing, just return the positive interaction
            return {
                'user_features': {
                    'user_id': torch.tensor([interaction['user_idx']])
                },
                'product_features': {
                    'product_id': torch.tensor([interaction['product_idx']])
                },
                'temporal_features': {
                    'hour': torch.tensor([interaction['temporal_features']['hour']]),
                    'day': torch.tensor([interaction['temporal_features']['day']]),
                    'month': torch.tensor([interaction['temporal_features']['month']]),
                    'days_since': torch.tensor([interaction['temporal_features']['days_since']])
                },
                'category_features': {
                    'department_ids': torch.tensor([interaction['department_idx']]),
                    'category_ids': torch.tensor([interaction['category_idx']])
                },
                'sequence_features': torch.tensor([interaction['sequence_products']]),
                'targets': torch.tensor([1.0])
            }