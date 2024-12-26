# src/inference/serving.py

from google.cloud import aiplatform
from google.cloud import storage
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import yaml
import time

class ModelServer:
    """Handles model serving and inference"""
    
    def __init__(self):
        # Load configs
        with open("config/endpoints.yaml") as f:
            endpoints_config = yaml.safe_load(f)["vertex_ai"]
            
        with open("config/config.yaml") as f:
            model_config = yaml.safe_load(f)["model"]["ncf"]
            
        # Initialize Vertex AI
        aiplatform.init(
            project=endpoints_config["project_id"],
            location=endpoints_config["region"]
        )
        
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{endpoints_config['project_id']}/locations/{endpoints_config['region']}/endpoints/{endpoints_config['endpoint_id']}"
        )
        
        # Load local model for embeddings
        self.model = self._load_local_model(model_config)
        self.model_version = model_config.get("version", "latest")
        
    def _load_local_model(self, config: Dict) -> torch.nn.Module:
        """Load local version of model for embedding generation"""
        try:
            # Download model from GCS if not present
            storage_client = storage.Client()
            bucket = storage_client.bucket(f"{config['project_id']}-model-artifacts")
            blob = bucket.blob("models/latest/model.pt")
            
            local_path = "/tmp/model.pt"
            blob.download_to_filename(local_path)
            
            # Load model
            model = torch.load(local_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
            
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