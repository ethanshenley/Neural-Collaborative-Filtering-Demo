# verify_deployment.py
from google.cloud import aiplatform
import yaml

def verify_deployment():
    # Load config
    with open("config/endpoints.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config['vertex_ai']['project_id'],
        location=config['vertex_ai']['region']
    )
    
    # Get endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=f"projects/{config['vertex_ai']['project_id']}/locations/{config['vertex_ai']['region']}/indexEndpoints/{config['vertex_ai']['endpoint_id']}"
    )
    
    # Print deployment status
    print("\nEndpoint Details:")
    print(f"Name: {endpoint.display_name}")
    print(f"Resource Name: {endpoint.resource_name}")
    print(f"Deployed Indexes: {endpoint.deployed_indexes}")
    
    return endpoint

if __name__ == "__main__":
    endpoint = verify_deployment()