# src/model/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from typing import Dict, Any, List
import logging
from tqdm import tqdm

from src.utils.metrics import calculate_metrics
from src.data.negative_sampler import NegativeSampler

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        num_gpus: int
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = num_gpus
        
        # Initialize distributed model if using multiple GPUs
        if num_gpus > 1:
            planner = EmbeddingShardingPlanner(
                topology=Topology(world_size=num_gpus),
                batch_size=config["batch_size"],
            )
            
            self.model = DistributedModelParallel(
                module=model,
                plan=planner.plan(model),
            )
        else:
            self.model = model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"]
        )
        self.criterion = nn.BCELoss()
        
        # Initialize negative sampler
        self.negative_sampler = NegativeSampler(
            num_products=config["num_products"],
            num_negative=config["num_negative_samples"]
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (features, targets) in enumerate(pbar):
                # Generate negative samples
                neg_features, neg_targets = self.negative_sampler.sample(
                    features, targets
                )
                
                # Combine positive and negative samples
                all_features = features.concat(neg_features)
                all_targets = torch.cat([targets, neg_targets])
                
                # Forward pass
                outputs = self.model(all_features)
                loss = self.criterion(outputs, all_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        return total_loss / len(train_loader)
    
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