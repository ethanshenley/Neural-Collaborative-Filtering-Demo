import torch
from google.cloud import bigquery
from google.cloud import storage
import logging
from pathlib import Path

from src.model.architecture import AdvancedNCF
from src.model.trainer import ModelTrainer
from src.utils.config import ConfigLoader

def main():
    # Load config
    config = ConfigLoader().get_config()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize BigQuery client
    bq_client = bigquery.Client()
    
    # Load enriched features
    logger.info("Loading features from BigQuery...")
    user_features_query = """
    SELECT * FROM `sheetz-poc.sheetz_data.user_features_enriched`
    """
    user_features_df = bq_client.query(user_features_query).to_dataframe()
    
    product_features_query = """
    SELECT * FROM `sheetz-poc.sheetz_data.product_features_enriched`
    """
    product_features_df = bq_client.query(product_features_query).to_dataframe()
    
    # Initialize model
    model = AdvancedNCF(
        num_users=len(user_features_df['cardnumber'].unique()),
        num_products=len(product_features_df['product_id'].unique()),
        num_departments=len(product_features_df['department_id'].unique()),
        num_categories=len(product_features_df['category_id'].unique()),
        mf_embedding_dim=config.model.embedding_dim,
        mlp_embedding_dim=config.model.embedding_dim,
        temporal_dim=32,
        mlp_hidden_dims=config.model.hidden_dims,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    # Initialize trainer with TorchRec distributed capabilities
    trainer = ModelTrainer(
        model=model,
        config=config.model,
        num_gpus=torch.cuda.device_count()
    )
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(
        user_features_df=user_features_df,
        product_features_df=product_features_df,
        batch_size=config.model.batch_size
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.model.num_epochs
    )
    
    # Save to GCS
    model_path = Path(config.model.save_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.gcp.bucket_name)
    blob = bucket.blob(f"models/{model_path.name}")
    blob.upload_from_filename(str(model_path))

if __name__ == "__main__":
    main()