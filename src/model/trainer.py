# src/model/trainer.py

import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology

from torchrec import KeyedJaggedTensor
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

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

        # Add this after initializing optimizer
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss for recommendations
        
        # Ensure criterion is on the same device as the model
        self.criterion = self.criterion.to(self.device)
        
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
        """Create training and validation data loaders"""
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
        
        # Verify first batch
        for features, targets in train_loader:
            # For KeyedJaggedTensor, we can get batch size from lengths
            batch_size = len(features.lengths()) // len(features.keys())
            if batch_size < 2:
                logging.warning(f"Small batch detected: {batch_size}")
            logging.info(f"First batch size: {batch_size}")
            break
                    
        return train_loader, val_loader

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

    def _validate_dimensions(self, tensor: torch.Tensor, expected_shape: tuple, name: str) -> None:
        """Utility to validate tensor dimensions"""
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}. "
                f"Expected: {expected_shape}, "
                f"Got: {tensor.shape}"
            )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with robust error handling and logging.
        
        Args:
            train_loader: DataLoader providing training batches
            
        Returns:
            Average loss for the epoch
            
        Raises:
            RuntimeError: If training fails
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress tracking
        with tqdm(train_loader, desc="Training", leave=True) as pbar:
            for batch_idx, (features, targets) in enumerate(pbar):
                try:
                    # Get base batch size from features
                    base_batch_size = len(features.lengths()) // len(features.keys())
                    effective_batch_size = base_batch_size * (1 + self.negative_samples)
                    
                    # Log batch info for first batch
                    if batch_idx == 0:
                        logging.info(
                            f"First batch - Base size: {base_batch_size}, "
                            f"Effective size: {effective_batch_size}"
                        )
                    
                    # Skip problematic batches
                    if base_batch_size < 2:
                        logging.warning(f"Skipping small batch {batch_idx}: size {base_batch_size}")
                        continue
                    
                    # Move tensors to correct device
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass with timing
                    start_time = time.time()
                    outputs = self.model(features)
                    forward_time = time.time() - start_time
                    
                    # Ensure matching shapes
                    if outputs.shape != targets.shape:
                        logging.debug(f"Reshaping tensors:")
                        logging.debug(f"- Outputs: {outputs.shape}")
                        logging.debug(f"- Targets: {targets.shape}")
                        
                        outputs = outputs.view(effective_batch_size, 1)
                        targets = targets.view(effective_batch_size, 1)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass with timing
                    start_time = time.time()
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping if configured
                    if hasattr(self.config, 'gradient_clipping'):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['gradient_clipping']
                        )
                    
                    self.optimizer.step()
                    backward_time = time.time() - start_time
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Calculate batch metrics
                    with torch.no_grad():
                        batch_accuracy = (outputs.round() == targets).float().mean()
                        pos_accuracy = (outputs[:base_batch_size].round() == targets[:base_batch_size]).float().mean()
                        
                        # This part changes:
                        if outputs.shape[0] > base_batch_size:
                            neg_accuracy = (outputs[base_batch_size:].round() == targets[base_batch_size:]).float().mean()
                        else:
                            neg_accuracy = torch.tensor(float('nan'))  # or skip computing neg_accuracy                    
                   
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_loss/num_batches:.4f}',
                        'acc': f'{batch_accuracy.item():.3f}',
                        'pos_acc': f'{pos_accuracy.item():.3f}',
                        'neg_acc': f'{neg_accuracy.item():.3f}' if not torch.isnan(neg_accuracy) else "N/A",
                        'fwd': f'{forward_time:.3f}s',
                        'bwd': f'{backward_time:.3f}s'
                    })
        
                    # Detailed logging every N batches
                    if batch_idx % 100 == 0:
                        self._log_training_stats(
                            batch_idx=batch_idx,
                            loss=loss.item(),
                            accuracy=batch_accuracy.item(),
                            pos_accuracy=pos_accuracy.item(),
                            neg_accuracy=neg_accuracy.item(),
                            forward_time=forward_time,
                            backward_time=backward_time
                        )
                
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}:")
                    logging.error(f"Feature keys: {features.keys()}")
                    logging.error(f"Target shape: {targets.shape}")
                    logging.error(f"Error details: {str(e)}")
                    raise
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f"Epoch complete - Average loss: {avg_loss:.4f}")
        
        return avg_loss

    def _log_training_stats(self, batch_idx: int, **metrics):
        """Log detailed training statistics.
        
        Args:
            batch_idx: Current batch index
            **metrics: Dictionary of metrics to log
        """
        logging.info(f"\nTraining Stats - Batch {batch_idx}:")
        for name, value in metrics.items():
            logging.info(f"- {name}: {value:.4f}")

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model on validation set.
        
        Args:
            val_loader: DataLoader providing validation batches
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validation", leave=False):
                try:
                    # Move to device
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(features)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Collect predictions
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                    
                except Exception as e:
                    logging.error("Error during validation:")
                    logging.error(f"Feature keys: {features.keys()}")
                    logging.error(f"Target shape: {targets.shape}")
                    logging.error(f"Error details: {str(e)}")
                    raise
        
        # Calculate metrics
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        metrics = calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        # Log validation results
        logging.info("\nValidation Results:")
        for name, value in metrics.items():
            logging.info(f"- {name}: {value:.4f}")
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 5,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Complete training loop with checkpointing and early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement
            checkpoint_dir: Directory to save checkpoints (optional)
            
        Returns:
            Dictionary containing training history
            
        Raises:
            RuntimeError: If training fails
        """
        # Initialize training state
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_hit_rate': [],
            'val_ndcg': [],
            'learning_rate': []
        }
        
        # Setup checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            latest_checkpoint = self._find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                start_epoch = self._load_checkpoint(latest_checkpoint)
                logging.info(f"Resuming training from epoch {start_epoch}")

        try:
            # Training loop
            for epoch in range(start_epoch, num_epochs):
                logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
                epoch_start_time = time.time()
                
                # Train epoch
                train_loss = self.train_epoch(train_loader)
                history['train_loss'].append(train_loss)
                
                # Validate
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_hit_rate'].append(val_metrics['hit_rate'])
                history['val_ndcg'].append(val_metrics['ndcg'])
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                self._log_epoch_results(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                    epoch_time=epoch_time
                )
                
                # Check for improvement
                val_loss = val_metrics['loss']
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        self._save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            epoch=epoch,
                            metrics=val_metrics,
                            is_best=True
                        )
                        logging.info(f"Saved new best model (improvement: {improvement:.4f})")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement for {patience_counter} epochs")
                    
                    # Save regular checkpoint
                    if checkpoint_dir:
                        self._save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            epoch=epoch,
                            metrics=val_metrics,
                            is_best=False
                        )
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Learning rate scheduling (if configured)
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(val_loss)
                    
            # Training complete
            logging.info("\nTraining completed:")
            logging.info(f"- Best validation loss: {best_val_loss:.4f}")
            logging.info(f"- Final learning rate: {current_lr:.6f}")
            
            return history
            
        except Exception as e:
            logging.error("Training failed:")
            logging.error(f"Error details: {str(e)}")
            
            # Save emergency checkpoint
            if checkpoint_dir:
                emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
                self._save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    metrics=val_metrics if 'val_metrics' in locals() else None,
                    is_best=False,
                    filename="emergency_checkpoint.pt"
                )
                logging.info(f"Saved emergency checkpoint to {emergency_path}")
            
            raise

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
        filename: Optional[str] = None
    ) -> None:
        """Save training checkpoint with all necessary state."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch+1}.pt"
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'model_config': {
                'num_users': self.model.num_users,
                'num_products': self.model.num_products,
                'embedding_dim': self.model.mf_embedding_dim
            }
        }
        
        # Add scheduler state if exists
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # If best model, create symlink
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(filename, best_path)

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training state from checkpoint."""
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if exists
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Verify config matches
        if checkpoint['config'] != self.config:
            logging.warning("Checkpoint config differs from current config!")
            self._log_config_differences(checkpoint['config'], self.config)
            
        return checkpoint['epoch'] + 1

    def _log_config_differences(self, checkpoint_config: Dict, current_config: Dict) -> None:
        """Log differences between checkpoint and current config."""
        logging.warning("Config differences:")
        for key in set(checkpoint_config.keys()) | set(current_config.keys()):
            if key not in checkpoint_config:
                logging.warning(f"- {key}: Missing in checkpoint config")
            elif key not in current_config:
                logging.warning(f"- {key}: Missing in current config")
            elif checkpoint_config[key] != current_config[key]:
                logging.warning(
                    f"- {key}: Checkpoint={checkpoint_config[key]}, "
                    f"Current={current_config[key]}"
                )