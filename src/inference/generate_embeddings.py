# generate_embeddings.py
import os
import json
import uuid
import yaml
import torch
import logging
import tempfile
import numpy as np
from typing import Dict, Any
from google.cloud import storage
from src.model.architecture import AdvancedNCF
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    required_keys = [
        "gcp.project_id",
        "gcp.staging_bucket",
        "gcp.location",
        "model.ncf.model_save_path"
    ]
    
    for key in required_keys:
        parts = key.split('.')
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Missing required config key: {key}")
            current = current[part]
    
    return config

def download_model(storage_client: storage.Client, bucket_name: str, model_path: str) -> str:
    """Download model from GCS."""
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(model_path)
    
    if not model_blob.exists():
        raise FileNotFoundError(
            f"Model file not found in GCS: gs://{bucket_name}/{model_path}"
        )

    local_model_path = "/tmp/local_model.pt"
    logger.info(f"Downloading model from gs://{bucket_name}/{model_path}")
    model_blob.download_to_filename(local_model_path)
    return local_model_path

def initialize_model(model_path: str) -> AdvancedNCF:
    """Initialize and load the model."""
    logger.info("Initializing model...")
    model = AdvancedNCF(
        num_users=8031,
        num_products=366,
        num_departments=5,
        num_categories=24,
        mf_embedding_dim=64,
        mlp_embedding_dim=64,
        temporal_dim=32,
        mlp_hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2,
        negative_samples=4
    )

    state_dict = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

def create_features_dict(
    product_id: str,
    category_id: str,
    department_id: str,
    model: AdvancedNCF,
    category_map: Dict[str, int],
    department_map: Dict[str, int]
) -> Dict[str, Any]:
    """Create feature dictionary for model input."""
    try:
        # Map string IDs to integers
        category_id_int = category_map.get(category_id, 0)
        department_id_int = department_map.get(department_id, 0)
        
        # Ensure IDs are within valid ranges
        category_id_int = min(max(0, category_id_int), model.num_categories - 1)
        department_id_int = min(max(0, department_id_int), model.num_departments - 1)
        
        # Convert product ID to index
        product_idx = int(product_id.lstrip('P'), 16) % model.num_products
        
        return {
            "product_features": KeyedJaggedTensor(
                keys=["user_id", "product_id"],
                values=torch.tensor([0, product_idx], dtype=torch.long),
                lengths=torch.tensor([1, 1], dtype=torch.long),
                offsets=torch.tensor([0, 1, 2], dtype=torch.long)
            ),
            "category_features": {
                "department_ids": torch.tensor([department_id_int], dtype=torch.long),
                "category_ids": torch.tensor([category_id_int], dtype=torch.long),
            },
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error creating features for product {product_id}: {e}")
        raise
    
def generate_embeddings(config_path: str = "config/config.yaml") -> None:
    """Main function to generate embeddings."""
    try:
        # Load configuration
        config = load_config(config_path)
        project_id = config["gcp"]["project_id"]
        staging_bucket = config["gcp"]["staging_bucket"]
        model_bucket_path = config["model"]["ncf"]["model_save_path"]

        # Initialize GCS client
        storage_client = storage.Client(project=project_id)
        
        # Download and initialize model
        local_model_path = download_model(storage_client, staging_bucket, model_bucket_path)
        model = initialize_model(local_model_path)
        logger.info("Model loaded successfully!")

        # Setup for embeddings generation
        embeddings_filename = f"product_embeddings_{uuid.uuid4()}.jsonl"
        local_output_file = f"/tmp/{embeddings_filename}"
        
        # Process product data
        bucket = storage_client.bucket(staging_bucket)
        exported_data_prefix = "exports/product_data_9f91aa14-2883-40b0-a811-aa557847e65a"
        
        blobs = list(bucket.list_blobs(prefix=exported_data_prefix))
        json_blobs = [b for b in blobs if b.name.endswith(".json")]
        
        if not json_blobs:
            raise ValueError(f"No JSON files found in gs://{staging_bucket}/{exported_data_prefix}")

        processed_products = set()
        total_count = 0
        error_count = 0

        # Create category and department mappings
        category_set = set()
        department_set = set()

        # First pass to collect unique values
        for jb in json_blobs:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_in:
                jb.download_to_filename(temp_in.name)
                temp_in_path = temp_in.name

            with open(temp_in_path, "r") as in_f:
                for line in in_f:
                    row = json.loads(line.strip())
                    if cat_id := row.get("category_id"):
                        category_set.add(cat_id)
                    if dept_id := row.get("department_id"):
                        department_set.add(dept_id)
            
            os.unlink(temp_in_path)

        # Create mappings
        category_map = {cat: idx for idx, cat in enumerate(sorted(category_set))}
        department_map = {dept: idx for idx, dept in enumerate(sorted(department_set))}

        logger.info(f"Found {len(category_map)} unique categories: {category_map}")
        logger.info(f"Found {len(department_map)} unique departments: {department_map}")

        with open(local_output_file, "w") as out_f:
            for jb in json_blobs:
                logger.info(f"Processing file: gs://{staging_bucket}/{jb.name}")
                
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_in:
                    jb.download_to_filename(temp_in.name)
                    temp_in_path = temp_in.name

                with open(temp_in_path, "r") as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        try:
                            row = json.loads(line.strip())
                            product_id = row.get("product_id")
                            
                            if not product_id or product_id in processed_products:
                                continue

                            features_dict = create_features_dict(
                                product_id=product_id,
                                category_id=row.get("category_id", "0"),
                                department_id=row.get("department_id", "0"),
                                model=model,
                                category_map=category_map,
                                department_map=department_map
                            )

                            with torch.no_grad():
                                emb_dict = model.get_product_embeddings(features_dict)
                                final_vec = emb_dict["mlp"][0].cpu().numpy()
                                # Normalize the embedding vector
                                final_vec = final_vec / np.linalg.norm(final_vec)
                                final_vec = final_vec.tolist()

                            out_record = {
                                "id": str(product_id),
                                "embedding": final_vec
                            }
                            out_f.write(json.dumps(out_record) + "\n")
                            
                            processed_products.add(product_id)
                            total_count += 1

                            if total_count % 100 == 0:
                                logger.info(f"Processed {total_count} products")

                        except Exception as e:
                            error_count += 1
                            logger.error(f"Error processing line {line_num}: {e}")
                            continue

                os.unlink(temp_in_path)

        logger.info(f"Generated embeddings for {total_count} products with {error_count} errors")

        # Upload results to GCS
        output_blob_name = f"embeddings/{embeddings_filename}"
        blob_out = bucket.blob(output_blob_name)
        blob_out.upload_from_filename(local_output_file)

        logger.info(f"Embeddings uploaded to: gs://{staging_bucket}/{output_blob_name}")
        
        # Cleanup
        os.remove(local_output_file)
        os.remove(local_model_path)

    except Exception as e:
        logger.error(f"Fatal error in embeddings generation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    generate_embeddings("config/config.yaml")