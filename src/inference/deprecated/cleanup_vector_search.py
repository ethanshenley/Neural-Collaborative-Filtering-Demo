from google.cloud import aiplatform
import yaml

def cleanup_vector_search():
    """Clean up existing vector search resources"""
    try:
        # Load config
        with open("config/endpoints.yaml") as f:
            config = yaml.safe_load(f)
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config['vertex_ai']['project_id'],
            location=config['vertex_ai']['region']
        )

        # Get existing endpoint
        try:
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name="projects/726149969503/locations/us-central1/indexEndpoints/934105496539889664"
            )
            # Undeploy all indexes
            endpoint.undeploy_all()
            # Delete endpoint
            endpoint.delete()
            print("Cleaned up old endpoint")
        except Exception as e:
            print(f"No existing endpoint to clean up: {e}")

    except Exception as e:
        print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    cleanup_vector_search()