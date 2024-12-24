import torch
from google.cloud import bigquery
from google.cloud import storage
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import os
import pandas as pd

from src.model.architecture import AdvancedNCF
from src.model.trainer import ModelTrainer
from src.utils.config import ConfigLoader

def load_user_features(client: bigquery.Client, gcp_config: Dict) -> pd.DataFrame:
    """Load user features from BigQuery with error handling"""
    try:
        query = f"""
        SELECT * FROM `{gcp_config['project_id']}.{gcp_config['dataset_id']}.user_features_enriched`
        """
        df = client.query(query).to_dataframe()
        logging.info(f"Successfully loaded {len(df)} user features")
        return df
    except Exception as e:
        logging.error(f"Failed to load user features: {str(e)}")
        raise

def load_product_features(client: bigquery.Client, gcp_config: Dict) -> pd.DataFrame:
    """Load product features from BigQuery with error handling"""
    try:
        query = f"""
        SELECT * FROM `{gcp_config['project_id']}.{gcp_config['dataset_id']}.product_features_enriched`
        """
        df = client.query(query).to_dataframe()
        logging.info(f"Successfully loaded {len(df)} product features")
        return df
    except Exception as e:
        logging.error(f"Failed to load product features: {str(e)}")
        raise

def initialize_model(
    user_features_df: pd.DataFrame,
    product_features_df: pd.DataFrame,
    model_config: Dict,
) -> AdvancedNCF:
    """Initialize model with proper configuration validation"""
    try:
        required_params = ['embedding_dim', 'layers', 'num_heads', 'dropout']
        for param in required_params:
            if param not in model_config:
                raise ValueError(f"Missing required model parameter: {param}")

        model = AdvancedNCF(
            num_users=len(user_features_df['cardnumber'].unique()),
            num_products=len(product_features_df['product_id'].unique()),
            num_departments=len(product_features_df['department_id'].unique()),
            num_categories=len(product_features_df['category_id'].unique()),
            mf_embedding_dim=model_config['embedding_dim'],
            mlp_embedding_dim=model_config['embedding_dim'],
            temporal_dim=model_config.get('temporal_dim', 32),
            mlp_hidden_dims=model_config['layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            negative_samples=model_config.get('negative_samples', 4)  # Add this line
        )
        return model
    except Exception as e:
        logging.error(f"Failed to initialize model: {str(e)}")
        raise

def save_model_artifacts(
    model: AdvancedNCF,
    history: Dict,
    model_config: Dict,
    gcp_config: Dict,
    job_id: Optional[str] = None
) -> None:
    """Save model and training history to GCS"""
    try:
        # Create local paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_id = job_id or f"train_{timestamp}"
        model_dir = Path(model_config.get('model_dir', '/tmp/models'))
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{job_id}_model.pt"
        history_path = model_dir / f"{job_id}_history.pt"

        # Save locally
        torch.save(model.state_dict(), model_path)
        torch.save(history, history_path)
        logging.info(f"Saved model artifacts locally to {model_dir}")

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcp_config['staging_bucket'])

        model_blob = bucket.blob(f"models/{job_id}/{model_path.name}")
        model_blob.upload_from_filename(str(model_path))

        history_blob = bucket.blob(f"models/{job_id}/{history_path.name}")
        history_blob.upload_from_filename(str(history_path))
        
        logging.info(f"Uploaded model artifacts to gs://{gcp_config['staging_bucket']}/models/{job_id}/")
    except Exception as e:
        logging.error(f"Failed to save model artifacts: {str(e)}")
        raise

def build_trainer_config(
    gcp_config: Dict,
    training_config: Dict,
    user_features_df: pd.DataFrame,
    product_features_df: pd.DataFrame,
) -> Dict:
    """Build complete trainer configuration with validation"""
    num_users = len(user_features_df['cardnumber'].unique())
    num_products = len(product_features_df['product_id'].unique())
    
    if num_users == 0 or num_products == 0:
        raise ValueError(
            f"Invalid feature counts: users={num_users}, products={num_products}. "
            "Check BigQuery tables for data."
        )

    config = {
        # GCP settings
        'project_id': gcp_config['project_id'],
        'dataset_id': gcp_config['dataset_id'],
        
        # Model dimensions
        'num_users': num_users,
        'num_products': num_products,
        
        # Training parameters
        'batch_size': training_config['batch_size'],
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config.get('weight_decay', 1e-5),
        'epochs': training_config['epochs'],
        'num_workers': training_config.get('num_workers', 4),
        'validation_days': training_config.get('validation_days', 10),
        
        # Additional model parameters
        'embedding_dim': training_config['embedding_dim'],
        'temporal_dim': training_config.get('temporal_dim', 32),
        'num_heads': training_config['num_heads'],
        'dropout': training_config['dropout']
    }
    
    logging.info(f"Built trainer config with {num_users} users and {num_products} products")
    return config

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")

    try:
        # Initialize configs
        config_loader = ConfigLoader()
        gcp_config = config_loader.get_gcp_config()
        model_config = config_loader.get_model_config()
        data_config = config_loader.get_data_generation_config()

        # Initialize BigQuery client
        bq_client = bigquery.Client(project=gcp_config['project_id'])
        logger.info(f"Connected to BigQuery project: {gcp_config['project_id']}")

        # Load features with proper error handling
        try:
            user_features_df = load_user_features(bq_client, gcp_config)
            logger.info(f"Successfully loaded {len(user_features_df)} user features")
            
            product_features_df = load_product_features(bq_client, gcp_config)
            logger.info(f"Successfully loaded {len(product_features_df)} product features")
        except Exception as e:
            logger.error(f"Failed to load features: {str(e)}")
            raise

        # Build trainer config with validation
        trainer_config = build_trainer_config(
            gcp_config=gcp_config,
            training_config=model_config['ncf'],
            user_features_df=user_features_df,
            product_features_df=product_features_df
        )

        # Log configuration details
        logger.info("Trainer configuration:")
        for key, value in trainer_config.items():
            logger.info(f"  {key}: {value}")

        # Initialize model
        model = initialize_model(
            user_features_df=user_features_df,
            product_features_df=product_features_df,
            model_config=model_config['ncf']
        )
        logger.info("Model initialized successfully")

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            config=trainer_config,
            num_gpus=torch.cuda.device_count()
        )
        logger.info(f"Trainer initialized with {torch.cuda.device_count()} GPUs")

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = trainer.create_data_loaders(
            user_features_df=user_features_df,
            product_features_df=product_features_df,
            batch_size=trainer_config['batch_size'],
            num_workers=trainer_config.get('num_workers', 4),
            validation_days=trainer_config.get('validation_days', 10)
        )
        logger.info("Data loaders created successfully")

        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=trainer_config['epochs']
        )

        # Save model artifacts
        save_model_artifacts(
            model=model,
            history=history,
            model_config=model_config['ncf'],
            gcp_config=gcp_config,
            job_id=os.getenv('AIP_MODEL_DIR', '').split('/')[-1]
        )

        # Log final training metrics
        if history and 'val_metrics' in history:
            logger.info("Final validation metrics:")
            for metric, value in history['val_metrics'][-1].items():
                logger.info(f"  {metric}: {value:.4f}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()