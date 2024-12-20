import torch
from google.cloud import bigquery
from google.cloud import storage
import logging
from pathlib import Path

from src.model.architecture import AdvancedNCF
from src.model.trainer import ModelTrainer
from src.utils.config import ConfigLoader

def main():
    # Initialize configs
    config_loader = ConfigLoader()
    gcp_config = config_loader.get_gcp_config()
    model_config = config_loader.get_model_config()
    data_config = config_loader.get_data_generation_config()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")
    
    try:
        # Initialize BigQuery client
        bq_client = bigquery.Client(project=gcp_config['project_id'])
        logger.info(f"Connected to BigQuery project: {gcp_config['project_id']}")
        
        # Load enriched features
        logger.info("Loading features from BigQuery...")
        user_features_query = f"""
        SELECT * FROM `{gcp_config['project_id']}.{gcp_config['dataset_id']}.user_features_enriched`
        """
        user_features_df = bq_client.query(user_features_query).to_dataframe()
        logger.info(f"Loaded {len(user_features_df)} user features")
        
        product_features_query = f"""
        SELECT * FROM `{gcp_config['project_id']}.{gcp_config['dataset_id']}.product_features_enriched`
        """
        product_features_df = bq_client.query(product_features_query).to_dataframe()
        logger.info(f"Loaded {len(product_features_df)} product features")
        
        # Initialize model
        logger.info("Initializing model...")
        model = AdvancedNCF(
            num_users=len(user_features_df['cardnumber'].unique()),
            num_products=len(product_features_df['product_id'].unique()),
            num_departments=len(product_features_df['department_id'].unique()),
            num_categories=len(product_features_df['category_id'].unique()),
            mf_embedding_dim=model_config['ncf']['embedding_dim'],
            mlp_embedding_dim=model_config['ncf']['embedding_dim'],
            temporal_dim=model_config['ncf'].get('temporal_dim', 32),
            mlp_hidden_dims=model_config['ncf']['layers'],
            num_heads=model_config['ncf']['num_heads'],
            dropout=model_config['ncf']['dropout']
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            config=model_config['ncf'],
            num_gpus=torch.cuda.device_count()
        )
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(
            user_features_df=user_features_df,
            product_features_df=product_features_df,
            batch_size=model_config['ncf']['batch_size'],
            num_workers=model_config['ncf']['num_workers'],
            validation_days=model_config['ncf']['validation_days']
        )
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=model_config['ncf']['epochs']
        )
        
        # Save model and history
        logger.info("Saving model artifacts...")
        model_path = Path(model_config['ncf']['model_save_path'])
        history_path = Path(model_config['ncf']['history_save_path'])
        
        # Save locally first
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        torch.save(history, history_path)
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcp_config['staging_bucket'])
        
        model_blob = bucket.blob(f"models/{model_path.name}")
        model_blob.upload_from_filename(str(model_path))
        
        history_blob = bucket.blob(f"models/{history_path.name}")
        history_blob.upload_from_filename(str(history_path))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()