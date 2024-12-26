# scripts/inference/03_setup_vector_search.py

from google.cloud import aiplatform
from google.cloud import bigquery
import numpy as np
import logging
import torch

def setup_vector_search(
    project_id: str,
    location: str,
    index_id: str,
    dimensions: int
):
    """Setup Vertex AI Vector Search index"""
    aiplatform.init(project=project_id, location=location)
    
    # Create index
    index = aiplatform.MatchingEngineIndex.create(
        display_name=index_id,
        contents_delta_uri=f"gs://{project_id}-vertex-index",
        dimensions=dimensions,
        approximate_neighbors_count=100,
        distance_measure_type="DOT_PRODUCT",
        description="Product embedding index for recommendations"
    )
    
    # Deploy index
    index_endpoint = index.deploy(
        deployed_index_id=f"{index_id}-deployed",
        min_replica_count=1,
        max_replica_count=2
    )
    
    return index, index_endpoint

def create_product_embeddings():
    """Generate product embeddings using the trained model"""
    # Load model
    model = torch.load("gs://sheetz-rec-staging/models/best_model.pt")
    model.eval()
    
    # Get products from BigQuery
    client = bigquery.Client()
    query = """
    SELECT * FROM `sheetz_data.product_features_enriched`
    """
    products_df = client.query(query).to_dataframe()
    
    # Generate embeddings
    embeddings = []
    with torch.no_grad():
        for _, product in products_df.iterrows():
            embedding = model.get_product_embedding(product)
            embeddings.append(embedding.numpy())
    
    return products_df, np.array(embeddings)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Setup vector search
    index, endpoint = setup_vector_search(
        project_id="sheetz-poc",
        location="us-central1",
        index_id="product-embeddings",
        dimensions=64  # Match your embedding dimension
    )
    
    # Generate and upload embeddings
    products_df, embeddings = create_product_embeddings()
    
    # Update index with embeddings
    index.upsert_embeddings(
        embeddings=embeddings,
        ids=[str(i) for i in range(len(embeddings))]
    )
    
    logging.info("Vector search setup complete!")
    logging.info(f"Index endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    main()