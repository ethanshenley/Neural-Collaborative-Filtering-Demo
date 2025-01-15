import logging
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_static_tree_ah_index(
    gcs_embeddings_uri: str,
    display_name: str,
    dimensions: int,
    distance_measure_type: str = "COSINE_DISTANCE",
):
    """
    Create a Tree-AH index from a JSON lines file in GCS.
    
    Args:
        gcs_embeddings_uri: URI like "gs://my-bucket/my-embeddings.jsonl"
        display_name: Display name for the index resource
        dimensions: Dimensionality of each embedding
        distance_measure_type: e.g. "COSINE_DISTANCE"
    
    Returns:
        MatchingEngineIndex object
    """
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=gcs_embeddings_uri,
        dimensions=dimensions,
        distance_measure_type="COSINE_DISTANCE",
        approximate_neighbors_count=100,
        leaf_nodes_to_search_percent=10,
    )
    logger.info(f"Created Tree-AH index: {index.resource_name}")
    return index

def create_and_deploy_endpoint(
    index: aiplatform.MatchingEngineIndex,
    endpoint_display_name: str,
    deployed_index_id: str,
    public_endpoint_enabled: bool = True
) -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Create a new Matching Engine Index Endpoint and deploy the given index to it.
    
    Args:
        index: The MatchingEngineIndex object
        endpoint_display_name: Display name for the index endpoint
        deployed_index_id: A unique ID for the deployed index
        public_endpoint_enabled: Whether to enable a public endpoint
    
    Returns:
        MatchingEngineIndexEndpoint object
    """
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=endpoint_display_name,
        public_endpoint_enabled=public_endpoint_enabled
    )
    logger.info(f"Created index endpoint: {endpoint.resource_name}")

    # Deploy the index
    deploy_op = endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
    )
    deploy_op.wait()  # Blocks until deployment is complete

    logger.info(f"Index deployed successfully to endpoint with ID: {deployed_index_id}")
    return endpoint

def main():
    """
    1) Creates a Tree-AH index from your product embeddings in GCS.
    2) Deploys that index to a new endpoint.
    3) Demonstrates a basic inference (top-k neighbor search) using the endpoint.
    """
    # ---------------------------
    # 0) Configure your environment
    # ---------------------------
    project_id = "sheetz-poc"  # <---- Change this
    location = "us-central1"   # <---- Your region
    aiplatform.init(project=project_id, location=location)

    # Path to your JSON lines file in GCS containing embeddings:
    gcs_embeddings_uri = "gs://sheetz-rec-staging/embeddings/product_embeddings_5ed0622b-4a07-4cdf-bace-7c81303dae38.jsonl"

    index_display_name = "sheetz-product-tree-ah-index"
    endpoint_display_name = "sheetz-recommendation-endpoint"
    deployed_index_id = "sheetz_product_index_v5"

    # Embedding dimension
    dims = 64

    # ---------------------------
    # 1) Create the Tree-AH index
    # ---------------------------
    index = create_static_tree_ah_index(
        gcs_embeddings_uri=gcs_embeddings_uri,
        display_name=index_display_name,
        dimensions=dims,
        distance_measure_type="COSINE_DISTANCE"  
        # or "COSINE_DISTANCE", "SQUARED_L2_DISTANCE"
    )

    # ---------------------------
    # 2) Create & deploy endpoint
    # ---------------------------
    endpoint = create_and_deploy_endpoint(
        index=index,
        endpoint_display_name=endpoint_display_name,
        deployed_index_id=deployed_index_id,
        public_endpoint_enabled=True
    )

    # ---------------------------
    # 3) Simple "inference" demo
    # ---------------------------
    import numpy as np

    test_embedding = [-0.0031329842749983072, -0.0014982176944613457, -0.25768914818763733, -0.009149184450507164, 0.0598788745701313, -0.10726749897003174, 0.003819603705778718, -0.19996939599514008, -0.005111176520586014, 0.018150735646486282, 0.004002606961876154, 0.175477996468544, -0.1774701476097107, -0.07286956906318665, 0.08803778886795044, 0.044280603528022766, 0.01994629018008709, -0.010852252133190632, -0.02145182341337204, -0.04672422260046005, 0.03250851482152939, -0.10445000976324081, 0.011446475982666016, -0.0081477090716362, -0.09227349609136581, 0.024155298247933388, -0.014880489557981491, -0.13436520099639893, 0.009811908937990665, 0.04316897317767143, -0.027513664215803146, -0.08558830618858337, -0.016603782773017883, 0.04591191187500954, 0.10342945903539658, -0.0729534849524498, -0.07765236496925354, -0.0036520264111459255, -0.0006731279427185655, 0.10948114842176437, 0.08423049747943878, -0.23668469488620758, -0.01158906240016222, -0.08769828826189041, -0.2852933704853058, 0.3467768430709839, -0.3770122230052948, 0.08210732787847519, 0.042924314737319946, -0.007210886105895042, 0.020073302090168, -0.00784899853169918, 0.0071702999994158745, 0.12418831884860992, 0.05412992835044861, -0.05828312039375305, 0.12262272089719772, 0.0073631759732961655, -0.023873891681432724, 0.07352828234434128, 0.47583019733428955, 0.01965801790356636, 0.12724342942237854, 0.04322732612490654]

    # Call find_neighbors
    response = endpoint.find_neighbors(
        queries=[test_embedding],  # Just pass the test_embedding as is
        deployed_index_id=deployed_index_id,
        num_neighbors=5
    )
    
    print("Full response:", response)
    if response:
        print("First query neighbors:", response[0])
        if response[0]:
            print("First neighbor item:", response[0][0])
            print("Type of first neighbor item:", type(response[0][0]))


    # 'response' is presumably a list (one element per query). Let's get the first query's neighbors:
    neighbors_for_query0 = response[0]  # a list of neighbor objects or dicts

    print("Top-5 neighbors for user_vec:")
    for neighbor in neighbors_for_query0:
        # If neighbor is an object with .id, .distance:
        print(f"Product ID: {neighbor.id}, Distance: {neighbor.distance}")

        # If it's a dict, do:
        # print(f"Product ID: {neighbor['id']}, Distance: {neighbor['distance']}")

    print("Done! Integrate this endpoint into your recommendation flow.")



if __name__ == "__main__":
    main()