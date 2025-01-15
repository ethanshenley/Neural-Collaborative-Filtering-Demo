from google.cloud import aiplatform
import numpy as np
import yaml
import time

def test_vector_search():
    """Test the deployed vector search index"""
    
    # Load config
    with open("config/endpoints.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config['vertex_ai']['project_id'],
        location=config['vertex_ai']['region']
    )

    # Get the deployed endpoint
    endpoint_id = "934105496539889664"  # From your deployment output
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=f"projects/{config['vertex_ai']['project_id']}/locations/{config['vertex_ai']['region']}/indexEndpoints/{endpoint_id}"
    )

    # Create test datapoints for upsert
    dimensions = config['vector_search']['dimensions']
    num_test_points = 5
    
    datapoints = []
    for i in range(num_test_points):
        datapoints.append({
            "id": f"test_product_{i}",
            "embedding": np.random.rand(dimensions).tolist(),
            "restricts": {
                "product_id": f"test_product_{i}",
                "category": "test_category",
                "price": 9.99
            }
        })

    # Get the index
    index_id = "4172826947316875264"  # From your deployment output
    index = aiplatform.MatchingEngineIndex(
        index_name=f"projects/{config['vertex_ai']['project_id']}/locations/{config['vertex_ai']['region']}/indexes/{index_id}"
    )

    try:
        # Upsert data points
        print("Upserting test datapoints...")
        index.upsert_datapoints(
            datapoints=[{
                "id": dp["id"],
                "embedding": dp["embedding"],
                "restricts": dp["restricts"]
            } for dp in datapoints],
            sync=True
        )
        print("Upsert successful!")

        # Wait a bit for the upsert to propagate
        time.sleep(10)

        # Test query
        print("\nTesting vector search query...")
        query_vector = np.random.rand(dimensions).tolist()
        response = endpoint.find_neighbors(
            deployed_index_id="sheetz_rec_index_v1",
            queries=[query_vector],
            num_neighbors=3
        )
        
        print("\nQuery results:")
        for i, neighbors in enumerate(response):
            print(f"\nResults for query {i}:")
            for neighbor in neighbors:
                print(f"ID: {neighbor.id}, Distance: {neighbor.distance:.4f}")
                
        return True

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting vector search test...")
    success = test_vector_search()
    if success:
        print("\nVector search test completed successfully!")
    else:
        print("\nVector search test failed!")