import os
import torch
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import yaml
import logging
from typing import Dict, Any

from src.model.architecture import AdvancedNCF
from src.model.data_prep import create_data_loaders
from src.utils.metrics import calculate_metrics

def train_evaluate(args):
    """Main training function for Vertex AI"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training job with args: {args}")
    
    # Load features from BigQuery
    bq_client = bigquery.Client()
    
    # Load enriched user features
    user_features_query = f"""
    SELECT * FROM `{args.project_id}.{args.dataset_id}.user_features_enriched`
    """
    logger.info("Loading user features...")
    user_features_df = bq_client.query(user_features_query).to_dataframe()
    
    # Load enriched product features
    product_features_query = f"""
    SELECT * FROM `{args.project_id}.{args.dataset_id}.product_features_enriched`
    """
    logger.info("Loading product features...")
    product_features_df = bq_client.query(product_features_query).to_dataframe()
    
    # Get user interactions
    interactions_query = f"""
    SELECT
        thf.cust_code as user_id,
        tbf.inventory_code as product_id,
        tbf.extended_retail as amount,
        thf.physical_date_time as transaction_timestamp
    FROM `{args.project_id}.{args.dataset_id}.transaction_header_fact` thf
    JOIN `{args.project_id}.{args.dataset_id}.transaction_body_fact` tbf
        ON thf.store_number = tbf.store_number
        AND thf.transaction_number = tbf.transaction_number
    WHERE thf.cust_code IS NOT NULL
    ORDER BY thf.physical_date_time DESC
    """
    logger.info("Loading interactions...")
    interactions_df = bq_client.query(interactions_query).to_dataframe()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        interactions_df=interactions_df,
        user_features_df=user_features_df,
        product_features_df=product_features_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_days=args.validation_days,
        negative_samples=args.negative_samples
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = AdvancedNCF(
        num_users=len(user_features_df['user_id'].unique()),
        num_products=len(product_features_df['product_id'].unique()),
        num_departments=len(product_features_df['department_id'].unique()),
        num_categories=len(product_features_df['category_id'].unique()),
        mf_embedding_dim=args.embedding_dim,
        mlp_embedding_dim=args.embedding_dim,
        temporal_dim=32,
        mlp_hidden_dims=args.hidden_dims,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(args.device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCELoss()
    
    # Setup model checkpointing
    checkpoint_dir = os.path.join(args.model_dir, args.job_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validate
        model.eval()
        val_loss = 0
        val_metrics = []
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
                metrics = calculate_metrics(model, batch)
                val_metrics.append(metrics)
        
        # Average metrics
        avg_val_metrics = {
            k: sum(m[k] for m in val_metrics) / len(val_metrics)
            for k in val_metrics[0].keys()
        }
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss/len(train_loader):.4f}, "
            f"Val Loss = {val_loss/len(val_loader):.4f}, "
            f"HR@10 = {avg_val_metrics['hit_rate']:.4f}, "
            f"NDCG@10 = {avg_val_metrics['ndcg']:.4f}"
        )
        
        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': avg_val_metrics,
            }, model_path)
            
            # Copy to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(args.bucket_name)
            blob = bucket.blob(f"models/{args.job_name}/best_model.pt")
            blob.upload_from_filename(model_path)
            
    # Test final model
    logger.info("Evaluating on test set...")
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for batch in test_loader:
            metrics = calculate_metrics(model, batch)  
            test_metrics.append(metrics)
            
    avg_test_metrics = {
        k: sum(m[k] for m in test_metrics) / len(test_metrics)
        for k in test_metrics[0].keys()
    }
    
    logger.info(f"Test Metrics: {avg_test_metrics}")
    return avg_test_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Data params
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    
    # Model params 
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128, 64])
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--validation_days', type=int, default=10)
    parser.add_argument('--negative_samples', type=int, default=4) 
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Job info
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='/tmp/models')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_evaluate(args)
