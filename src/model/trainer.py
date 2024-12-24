# src/model/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec import KeyedJaggedTensor
from typing import Dict, Any, List, Tuple
import pandas as pd
from tqdm import tqdm
from src.model.data_prep import SheetzDataset, create_data_loaders, collate_recommender_batch, ConsistentBatchSampler
from src.utils.metrics import calculate_metrics
from src.data.negative_sampler import NegativeSampler

from google.api_core import retry, exceptions
from google.cloud import bigquery
import logging

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        num_gpus: int
    ):
        
         # Required configuration parameters
        required_params = {
            'num_users',
            'num_products',
            'batch_size',
            'learning_rate',
            'project_id',
            'dataset_id'
        }

        missing_params = required_params - set(config.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in config: {missing_params}")
       
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = num_gpus
        self.negative_samples = config.get('negative_samples', 4)
        self.logger = logging.getLogger(__name__)
        
        weight_decay = float(config.get('weight_decay', 1e-5))  # Explicit conversion to float
        
        # Validate required configuration
        required_params = {'num_products', 'num_users', 'batch_size', 'learning_rate'}
        missing_params = required_params - set(config.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters in config: {missing_params}")
        
        # Initialize negative sampler
        self.negative_sampler = NegativeSampler(
            num_products=config['num_products'],
            num_negative=config.get('num_negative_samples', 4)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=weight_decay
        )
        
        # Move model to appropriate device
        if num_gpus > 1:
            self.model = DistributedModelParallel(
                module=self.model,
                device_ids=list(range(num_gpus))
            )
        else:
            self.model = self.model.to(self.device)

        # Log configuration for debugging
        logging.info("Initializing ModelTrainer with config:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

    def create_data_loaders(
        self,
        user_features_df: pd.DataFrame,
        product_features_df: pd.DataFrame,
        batch_size: int = 256,
        num_workers: int = 4,
        validation_days: int = 10
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders
        
        Args:
            user_features_df: DataFrame with user features
            product_features_df: DataFrame with product features
            batch_size: Batch size for training
            num_workers: Number of worker processes
            validation_days: Number of days to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logging.info("Creating data loaders...")
        
        # Create datasets
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
        
        # Create samplers
        train_sampler = ConsistentBatchSampler(
            dataset_size=len(train_dataset),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_recommender_batch,
            batch_sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_recommender_batch
        )
        
        logging.info(f"Created train loader with {len(train_loader)} batches")
        logging.info(f"Created val loader with {len(val_loader)} batches")
        
        # Verify batch sizes
        for i, (features, targets) in enumerate(train_loader):
            batch_size = features.size(0)
            if batch_size < 2:
                logging.warning(f"Small batch detected in train loader: {batch_size}")
            if i == 0:
                logging.info(f"First batch size: {batch_size}")
            if i >= 5:  # Check first few batches
                break
                
        return train_loader, val_loader

    @staticmethod
    def collate_recommender_batch(batch):
        """Custom collate function for batching KeyedJaggedTensor samples"""
        features_list = []
        targets_list = []
        
        for features, targets in batch:
            features_list.append(features)
            targets_list.append(targets)
            
        # Concatenate KeyedJaggedTensors
        batched_features = KeyedJaggedTensor.concat(features_list)
        batched_targets = torch.cat(targets_list)
        
        return batched_features, batched_targets
    
    @retry.Retry(
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        predicate=retry.if_exception_type(
            (exceptions.ServerError,
             exceptions.Forbidden,
             exceptions.ServiceUnavailable)
        )
    )

    def _execute_bigquery(self, query: str) -> pd.DataFrame:
        """Execute BigQuery with retry logic"""
        client = bigquery.Client(project=self.config['project_id'])
        return client.query(query).to_dataframe()

    def create_interactions_df(self):
        """Create interaction data from transaction facts"""
        query = """
        SELECT
            thf.cust_code as user_id,
            tbf.inventory_code as product_id,
            tbf.extended_retail as amount,
            thf.physical_date_time as transaction_timestamp
        FROM `{project_id}.{dataset_id}.transaction_header_fact` thf
        JOIN `{project_id}.{dataset_id}.transaction_body_fact` tbf
            ON thf.store_number = tbf.store_number
            AND thf.transaction_number = tbf.transaction_number
        WHERE thf.cust_code IS NOT NULL
        ORDER BY thf.physical_date_time DESC
        """.format(
            project_id=self.config['project_id'],
            dataset_id=self.config['dataset_id']
        )
        
        try:
            df = self._execute_bigquery(query)
            logging.info(f"Successfully loaded {len(df)} interactions")
            return df
        except Exception as e:
            logging.error(f"Failed to load interactions after retries: {str(e)}")
            raise

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (features, targets) in enumerate(pbar):
                # Check batch size using KeyedJaggedTensor methods
                batch_size = len(features.values()) // len(features.keys())
                if batch_size < 2:
                    logging.warning(f"Small batch detected: {batch_size}")
                    continue
                    
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                try:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                except Exception as e:
                    logging.error(f"Error in forward pass: {str(e)}")
                    logging.error(f"Batch size: {batch_size}")
                    logging.error(f"Feature keys: {features.keys()}")
                    raise
                    
                # Rest of training logic...                    
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'batch_size': features.size(0)
                })
                
        return total_loss / num_batches if num_batches > 0 else float('inf')
      
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation"):
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Calculate metrics
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        metrics = calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """Complete training loop."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            
            # Log metrics
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val HR@10: {val_metrics['hit_rate']:.4f}, "
                f"Val NDCG@10: {val_metrics['ndcg']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 
                         self.config["model_save_path"])
                self.logger.info("Saved best model checkpoint")
        
        return history