# src/data/data_prep.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

import logging

class SheetzDataset(Dataset):
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        product_features_df: pd.DataFrame,
        mode: str = 'train',
        validation_days: int = 10,
        negative_samples: int = 4
    ):
        """Initialize SheetzDataset.
        
        Args:
            interactions_df: DataFrame with user-product interactions
            user_features_df: DataFrame with user features
            product_features_df: DataFrame with product features
            mode: Either 'train' or 'val'
            validation_days: Number of days to use for validation
            negative_samples: Number of negative samples per positive interaction
        """
        self.mode = mode
        self.negative_samples = negative_samples
        
        # Add column validation with comprehensive checks
        required_cols = {
            'interactions': ['user_id', 'product_id', 'amount', 'transaction_timestamp'],
            'user_features': ['cardnumber', 'recent_interactions', 'preferred_categories'],
            'product_features': ['product_id', 'total_purchases', 'total_revenue']
        }
        
        # Validate columns exist with detailed error messages
        for df_name, cols in required_cols.items():
            df = {
                'interactions': interactions_df,
                'user_features': user_features_df,
                'product_features': product_features_df
            }[df_name]
            
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"{df_name} DataFrame missing required columns: {missing_cols}.\n"
                    f"Available columns: {df.columns.tolist()}"
                )
        
        # Log dataset sizes
        logging.info(f"Initializing dataset with:")
        logging.info(f"- {len(interactions_df)} interactions")
        logging.info(f"- {len(user_features_df)} unique users")
        logging.info(f"- {len(product_features_df)} unique products")
        
        # Create mappings using correct column names
        self.user_to_idx = {
            user: idx for idx, user in enumerate(user_features_df['cardnumber'].unique())
        }
        
        self.product_to_idx = {
            prod: idx for idx, prod in enumerate(product_features_df['product_id'].unique())
        }
        
        # Store dimensions for later use
        self.num_users = len(self.user_to_idx)
        self.num_products = len(self.product_to_idx)
        
        # Split data based on time
        latest_date = pd.to_datetime(interactions_df['transaction_timestamp']).max()
        split_date = latest_date - pd.Timedelta(days=validation_days)
        
        if mode == 'train':
            self.interactions = interactions_df[
                pd.to_datetime(interactions_df['transaction_timestamp']) < split_date
            ].copy()
        else:
            self.interactions = interactions_df[
                pd.to_datetime(interactions_df['transaction_timestamp']) >= split_date
            ].copy()
            
        # Create interaction list
        self.interaction_list = self._create_interaction_list()
        logging.info(f"Created {len(self.interaction_list)} valid interactions")
        
        # Calculate product weights for negative sampling
        product_counts = np.zeros(self.num_products)
        for _, product_idx, _ in self.interaction_list:
            product_counts[product_idx] += 1
            
        # Use inverse popularity for negative sampling (less popular items more likely to be negative)
        product_counts = np.maximum(product_counts, 1)  # Avoid division by zero
        self.product_weights = 1 / product_counts
        self.product_weights = self.product_weights / self.product_weights.sum()  # Normalize
        
        logging.info("Initialized product weights for negative sampling")
        
        # Initialize user-product history for negative sampling
        if mode == 'train':
            self._create_user_product_history()
            logging.info(f"Created user-product history for {len(self.user_product_history)} users")  
    
    def _create_interaction_list(self) -> List[Tuple[int, int, float]]:
        """Create list of (user_idx, product_idx, amount) tuples"""
        interactions = []
        
        # Add debug logging
        logging.info(f"Creating interaction list from {len(self.interactions)} interactions")
        logging.info(f"Available columns: {self.interactions.columns.tolist()}")
        
        for _, row in self.interactions.iterrows():
            # Use consistent column names from our query
            user_idx = self.user_to_idx.get(row['user_id'])  # Changed from cust_code
            product_idx = self.product_to_idx.get(row['product_id'])  # Changed from inventory_code
            
            # Skip if user or product not found in mappings
            if user_idx is None or product_idx is None:
                continue
                
            amount = row['amount']  # Changed from extended_retail
            interactions.append((user_idx, product_idx, amount))
        
        logging.info(f"Created {len(interactions)} valid interactions")
        return interactions  
          
    def _sample_negative(self, user_idx: int, positive_product: int) -> int:
        """Sample a negative product for a user, avoiding positive interactions"""
        max_attempts = 10  # Prevent infinite loop
        user_positives = set()
        
        # Get user's positive interactions if they exist
        if hasattr(self, 'user_product_history'):
            user_positives = self.user_product_history.get(user_idx, set())
        
        for _ in range(max_attempts):
            # Sample based on inverse popularity
            product_idx = np.random.choice(
                self.num_products,
                p=self.product_weights
            )
            # Ensure we don't sample the positive product or other positives
            if product_idx != positive_product and product_idx not in user_positives:
                return product_idx
                    
        # Fallback: randomly select from products not in user history
        valid_negatives = list(set(range(self.num_products)) - user_positives - {positive_product})
        if not valid_negatives:
            # If no valid negatives (rare case), just return a random product different from positive
            while True:
                idx = np.random.randint(0, self.num_products)
                if idx != positive_product:
                    return idx
        return np.random.choice(valid_negatives)
    
    def _create_user_product_history(self):
        """Create dictionary of user-product interactions with memory optimization"""
        self.user_product_history = {}
        chunk_size = 1000000  # Process in chunks to manage memory
        
        for i in range(0, len(self.interaction_list), chunk_size):
            chunk = self.interaction_list[i:i + chunk_size]
            for user_idx, product_idx, _ in chunk:
                if user_idx not in self.user_product_history:
                    self.user_product_history[user_idx] = set()
                self.user_product_history[user_idx].add(product_idx)
                
            if i % chunk_size == 0:
                logging.info(f"Processed {i}/{len(self.interaction_list)} interactions")
                
    def __len__(self) -> int:
        return len(self.interaction_list)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a training/validation item with proper tensor shape handling.
        
        Args:
            idx: Index into interaction list
            
        Returns:
            Tuple of (features dict, target tensor)
        """
        user_idx, product_idx, amount = self.interaction_list[idx]
        
        if self.mode == 'train':
            try:
                # Sample negative products
                negative_products = []
                for _ in range(self.negative_samples):
                    neg_product = self._sample_negative(user_idx, product_idx)
                    negative_products.append(neg_product)
                    
                # Create sparse feature tensors
                user_indices = torch.tensor([user_idx] * (1 + self.negative_samples))
                product_indices = torch.tensor([product_idx] + negative_products)

                # Create features dictionary
                features = {
                    'user_id': user_indices,
                    'product_id': product_indices,
                }
                
                # Create binary target tensor (1 for positive, 0 for negatives)
                targets = torch.zeros(1 + self.negative_samples, dtype=torch.float32)
                targets[0] = 1.0
                
                return features, targets
                
            except Exception as e:
                logging.error(f"Error in __getitem__ for idx {idx}:")
                logging.error(f"User idx: {user_idx}, Product idx: {product_idx}")
                logging.error(f"Error details: {str(e)}")
                raise
                
        else:  # Validation mode
            # For validation, just return the single positive interaction
            features = {
                'user_id': torch.tensor([user_idx]),
                'product_id': torch.tensor([product_idx])
            }
            return features, torch.tensor([1.0], dtype=torch.float32)
                      
def collate_recommender_batch(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
) -> Tuple[KeyedJaggedTensor, torch.Tensor]:
    """
    Collate batch of samples so each user_id and product_id is a single-element bag.
    This allows EmbeddingBagCollection with pooling='SUM' (or 'MEAN') to effectively
    behave as if there's no pooling.
    """
    try:
        user_values = []
        user_lengths = []
        
        product_values = []
        product_lengths = []
        
        all_targets = []
        
        # For debugging shape issues
        sample_shapes = []
        
        # Loop over each item in the batch
        for features, targets in batch:
            # Quick checks
            if not isinstance(features, dict):
                raise ValueError(f"Expected dict features, got {type(features)}")
            if set(features.keys()) != {"user_id", "product_id"}:
                raise ValueError(f"Invalid feature keys: {features.keys()}")
            if not isinstance(targets, torch.Tensor):
                raise ValueError(f"Targets must be tensor, got {type(targets)}")

            # For logging/debugging
            sample_shapes.append({
                "user_id": features["user_id"].shape,
                "product_id": features["product_id"].shape,
                "targets": targets.shape,
            })
            
            # Suppose each sample has shape (M,) for user_id, product_id, and targets
            # M = 1 + negative_samples for train mode, or M=1 for val/test
            user_ids = features["user_id"]
            product_ids = features["product_id"]
            
            # Each ID is its own single-element bag => length=1
            for u in user_ids:
                user_values.append(u.item())
                user_lengths.append(1)
            
            for p in product_ids:
                product_values.append(p.item())
                product_lengths.append(1)
            
            # Same for targets: store them individually
            for t in targets:
                all_targets.append(t.item())
        
        # Convert to tensors
        user_values_t = torch.tensor(user_values, dtype=torch.long)
        user_lengths_t = torch.tensor(user_lengths, dtype=torch.long)
        
        product_values_t = torch.tensor(product_values, dtype=torch.long)
        product_lengths_t = torch.tensor(product_lengths, dtype=torch.long)
        
        # Build KeyedJaggedTensor
        # We have 2 keys: "user_id" and "product_id", each with as many single-element bags
        features_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["user_id", "product_id"],
            values=torch.cat([user_values_t, product_values_t], dim=0),
            lengths=torch.cat([user_lengths_t, product_lengths_t], dim=0),
        )
        
        # Build target tensor
        targets_t = torch.tensor(all_targets, dtype=torch.float32)
        if targets_t.dim() == 1:
            targets_t = targets_t.unsqueeze(1)  # shape => (N, 1)
        
        # Optional: sanity-check shapes if you want
        # e.g. check that the # of rows in `targets_t` matches total single-element IDs
        expected_num_ids = len(user_values)  # or len(product_values), they match
        if targets_t.size(0) != expected_num_ids:
            raise ValueError(
                f"Target mismatch: got {targets_t.size(0)} rows, expected {expected_num_ids}"
            )
        
        return features_kjt, targets_t

    except Exception as e:
        logging.error("Error in collate_recommender_batch:")
        logging.error(f"Batch size: {len(batch)}")
        logging.error(f"Sample shapes: {sample_shapes}")
        logging.error(f"Error details: {str(e)}")
        raise


def create_data_loaders(
    self,
    user_features_df: pd.DataFrame,
    product_features_df: pd.DataFrame, 
    batch_size: int = 256,
    num_workers: int = 4,
    validation_days: int = 10
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation and test data loaders."""
    
    # Create training dataset
    train_dataset = SheetzDataset(
        interactions_df=self.create_interactions_df(),
        user_features_df=user_features_df,
        product_features_df=product_features_df,
        mode='train',
        validation_days=validation_days,
        negative_samples=self.config.get('negative_samples', 4)
    )
    
    val_dataset = SheetzDataset(
        interactions_df=self.create_interactions_df(),
        user_features_df=user_features_df,
        product_features_df=product_features_df,
        mode='val',
        validation_days=validation_days
    )
    
    test_dataset = SheetzDataset(
        interactions_df=self.create_interactions_df(),
        user_features_df=user_features_df,
        product_features_df=product_features_df,
        mode='test',
        validation_days=validation_days
    )

    # Create sampler for training
    train_sampler = ConsistentBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_recommender_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_recommender_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_recommender_batch
    )
    
    logging.info(f"Created train loader with {len(train_loader)} batches")
    logging.info(f"Created val loader with {len(val_loader)} batches")
    logging.info(f"Created test loader with {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

class ConsistentBatchSampler:
    """Ensures consistent batch sizes by padding smaller batches"""
    
    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        """Initialize sampler with explicit parameters
        
        Args:
            dataset_size: Total number of samples in dataset
            batch_size: Desired batch size
            shuffle: Whether to shuffle samples
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate batching info
        self.num_batches = (dataset_size + batch_size - 1) // batch_size
        self.last_batch_size = dataset_size % batch_size
        
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
            
    def __iter__(self):
        # Create index array
        indices = list(range(self.dataset_size))
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Create batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.dataset_size)
            
            # Get batch indices
            batch_indices = indices[start_idx:end_idx]
            
            # Pad last batch if needed
            if len(batch_indices) < self.batch_size:
                padding_needed = self.batch_size - len(batch_indices)
                # Pad with repeated samples from the same batch
                padding_indices = batch_indices[:padding_needed]
                batch_indices.extend(padding_indices)
                
            yield batch_indices
            
    def __len__(self) -> int:
        """Return the number of batches"""
        return self.num_batches