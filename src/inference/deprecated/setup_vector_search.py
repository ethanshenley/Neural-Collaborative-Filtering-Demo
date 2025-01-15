from google.cloud import aiplatform
import yaml
from datetime import datetime

def create_vector_search():
    """Create an updatable HNSW index and endpoint."""
    try:
        # Load config
        with open("config/endpoints.yaml") as f:
            config = yaml.safe_load(f)
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config['vertex_ai']['project_id'],
            location=config['vertex_ai']['region']
        )

        # Generate unique IDs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        deployed_index_id = f"sheetz_rec_index_{timestamp}"

        # ---- CREATE an updatable HNSW index ----
        index = aiplatform.MatchingEngineIndex.create_hnsw_index(
            display_name=f"product_embeddings_{timestamp}",
            dimensions=config['vector_search']['dimensions'],  # e.g. 64
            approximate_neighbors_count=50,
            distance_measure_type="DOT_PRODUCT",               # or "COSINE" / "SQUARED_L2_DISTANCE"
            # For streaming updates:
            is_updatable=True,   
            shard_size=1,        # typical for small updatable indexes
        )
        print(f"Created HNSW index: {index.resource_name}")

        # Create endpoint
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=f"product_embeddings_endpoint_{timestamp}",
            public_endpoint_enabled=True
        )
        print(f"Created endpoint: {endpoint.resource_name}")

        # Deploy index
        deploy_op = endpoint.deploy_index(
            index=index,
            deployed_index_id=deployed_index_id
        )
        deploy_op.wait()

        print(f"Index deployed successfully with ID: {deployed_index_id}")

        # Save the new IDs to config
        config['vertex_ai']['index_id'] = deployed_index_id
        config['vertex_ai']['endpoint_id'] = endpoint.resource_name.split('/')[-1]

        with open("config/endpoints.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print("Updated config with new IDs")
        return index, endpoint

    except Exception as e:
        print(f"Error in vector search setup: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        index, endpoint = create_vector_search()
        print("Vector search setup completed successfully")
    except Exception as e:
        print(f"Setup failed: {str(e)}")
