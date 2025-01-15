# src/inference/serving.py
from google.oauth2 import service_account
import os
from google.cloud import aiplatform
from google.cloud import storage
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import yaml
import time
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelServer:
    def __init__(self):
        # Get environment variables
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.model_bucket = os.getenv('MODEL_BUCKET')
        self.model_version = os.getenv('MODEL_VERSION')
        self.endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

        if not all([self.project_id, self.model_bucket, self.endpoint_id, credentials_path]):
            raise ValueError("Missing required environment variables")

        # Initialize credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Initialize clients with explicit credentials
        self.storage_client = storage.Client(
            project=self.project_id,
            credentials=credentials
        )
        
        # Initialize Vertex AI with credentials
        aiplatform.init(
            project=self.project_id,
            location="us-central1",
            credentials=credentials
        )
        
        # Initialize the endpoint
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{self.project_id}/locations/us-central1/endpoints/{self.endpoint_id}"
        )
        
        # Load local model for embeddings
        self.model = self._load_local_model()

    def _load_local_model(self) -> torch.nn.Module:
        """Load local version of model for embedding generation"""
        try:
            model_path = f"models/{self.model_version}/{self.model_version}_model.pt"
            logging.info(f"Attempting to load model from: {model_path}")
            
            bucket = self.storage_client.bucket(self.model_bucket)
            blob = bucket.blob(model_path)
            
            if not blob.exists():
                blobs = list(bucket.list_blobs(prefix="models/"))
                available_paths = [b.name for b in blobs if b.name.endswith('.pt')]
                logging.error(f"Model not found at {model_path}. Available models: {available_paths}")
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            local_path = "/tmp/model.pt"
            logging.info(f"Downloading model to {local_path}")
            blob.download_to_filename(local_path)
            
            # Load state dict with weights_only=True
            state_dict = torch.load(local_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
            
            # Import correct model class
            from src.model.architecture import AdvancedNCF
            
            # Initialize model with correct dimensions from trained model
            model = AdvancedNCF(
                num_users=8031,              # Actual number of users
                num_products=366,            # Actual number of products
                num_departments=5,           # Actual number of departments
                num_categories=24,           # Actual number of categories
                mf_embedding_dim=64,         # Keep original
                mlp_embedding_dim=64,        # Keep original
                temporal_dim=32,             # Keep original
                mlp_hidden_dims=[256, 128, 64],  # Keep original
                num_heads=4,                 # Keep original
                dropout=0.2,                 # Keep original
                negative_samples=4           # Keep original
            )
            
            # Load the state dict into the model
            model.load_state_dict(state_dict)
            model.eval()
            
            logging.info("Model loaded successfully")
            return model
                
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise      

    async def get_predictions(
        self,
        user_features: Dict[str, Any],
        product_ids: List[str],
        batch_size: int = 100
    ) -> List[Dict[str, float]]:
        """Get prediction scores for user-product pairs"""
        try:
            start_time = time.time()
            
            # Prepare features for inference
            instances = []
            for product_id in product_ids:
                instance = {
                    "user_features": user_features,
                    "product_id": product_id
                }
                instances.append(instance)
                
            # Process in batches
            all_predictions = []
            for i in range(0, len(instances), batch_size):
                batch = instances[i:i + batch_size]
                
                # Get predictions from endpoint
                predictions = await self.endpoint.predict(instances=batch)
                all_predictions.extend(predictions.predictions)
                
            # Format results
            results = []
            for product_id, score in zip(product_ids, all_predictions):
                results.append({
                    "product_id": product_id,
                    "score": float(score[0])  # Assuming single score output
                })
                
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Log latency
            latency_ms = (time.time() - start_time) * 1000
            logging.info(f"Prediction latency: {latency_ms:.2f}ms for {len(product_ids)} products")
            
            return results
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise
            
    @torch.no_grad()
    def get_user_embedding(self, user_features: Dict[str, Any]) -> np.ndarray:
        """Generate user embedding for vector search"""
        try:
            # Convert features to tensor
            features = self._prepare_features(user_features)
            
            # Get embedding from model
            embedding = self.model.get_user_embedding(features)
            
            return embedding.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Embedding error: {str(e)}")
            raise
            
    def _prepare_features(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare features for model input"""
        prepared = {}
        
        # Convert numerical features
        for key in ["age", "lifetime_points", "account_age_days"]:
            if key in features:
                prepared[key] = torch.tensor([features[key]], dtype=torch.float32)
                
        # Convert categorical features
        for key in ["gender", "enrollment_status"]:
            if key in features:
                prepared[key] = torch.tensor([features[key]], dtype=torch.long)
                
        # Convert sequence
        if "sequence" in features:
            prepared["sequence"] = torch.tensor(features["sequence"], dtype=torch.long)
            
        # Convert temporal features
        if all(k in features for k in ["hour", "day", "days_since"]):
            prepared["temporal"] = torch.tensor([
                features["hour"],
                features["day"],
                features["days_since"]
            ], dtype=torch.float32)
            
        return prepared
        
    async def get_batch_predictions(
        self,
        user_features_batch: Dict[str, Dict[str, Any]],
        product_ids: List[str]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Get predictions for multiple users"""
        all_predictions = {}
        
        # Process each user
        for user_id, features in user_features_batch.items():
            predictions = await self.get_predictions(
                user_features=features,
                product_ids=product_ids
            )
            all_predictions[user_id] = predictions
            
        return all_predictions