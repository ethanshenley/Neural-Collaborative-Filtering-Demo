# src/inference/vector_search.py

from google.cloud import aiplatform
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import yaml

class ProductSearch:
    """Vector similarity search for product recommendations"""
    
    def __init__(self):
        # Load config
        with open("config/endpoints.yaml") as f:
            endpoints_config = yaml.safe_load(f)["vertex_ai"]
            
        # Initialize Vertex AI Vector Search
        self.index = aiplatform.MatchingEngineIndex(
            index_name=f"projects/{endpoints_config['project_id']}/locations/{endpoints_config['region']}/indexes/{endpoints_config['index_id']}"
        )
        
        # Initialize index endpoint
        self.index_endpoint = self.index.deploy(
            deployed_index_id=f"{endpoints_config['index_id']}-deployed"
        )
        
    async def find_neighbors(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find nearest neighbor products for an embedding"""
        try:
            # Apply filtering if category specified
            filter_str = None
            if category_filter:
                filter_str = f'category_id = "{category_filter}"'
                
            # Query index
            response = await self.index_endpoint.find_neighbors_async(
                queries=[query_embedding],
                num_neighbors=k,
                filter=filter_str
            )
            
            # Process results
            results = []
            for match in response.nearest_neighbors[0]:
                results.append({
                    "product_id": match.id,
                    "score": float(match.distance)  # Cosine similarity score
                })
                
            return results
            
        except Exception as e:
            logging.error(f"Vector search error: {str(e)}")
            raise
            
    async def batch_find_neighbors(
        self,
        query_embeddings: List[np.ndarray],
        k: int = 10,
        category_filter: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """Batch search for multiple query embeddings"""
        try:
            filter_str = None
            if category_filter:
                filter_str = f'category_id = "{category_filter}"'
                
            # Batch query
            response = await self.index_endpoint.find_neighbors_async(
                queries=query_embeddings,
                num_neighbors=k,
                filter=filter_str
            )
            
            # Process results
            all_results = []
            for neighbors in response.nearest_neighbors:
                results = []
                for match in neighbors:
                    results.append({
                        "product_id": match.id,
                        "score": float(match.distance)
                    })
                all_results.append(results)
                
            return all_results
            
        except Exception as e:
            logging.error(f"Batch vector search error: {str(e)}")
            raise
            
    async def update_index(
        self,
        embeddings: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, Any]]
    ):
        """Update product embeddings in the index"""
        try:
            # Prepare index data
            index_data = []
            for product_id, embedding in embeddings.items():
                product_metadata = metadata.get(product_id, {})
                
                index_data.append({
                    "id": product_id,
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "category_id": product_metadata.get("category_id", ""),
                        "department_id": product_metadata.get("department_id", ""),
                        "last_modified": product_metadata.get("last_modified_date", "")
                    }
                })
            
            # Update index in batches
            batch_size = 100
            for i in range(0, len(index_data), batch_size):
                batch = index_data[i:i + batch_size]
                
                await self.index.upsert_async(
                    embeddings=[item["embedding"] for item in batch],
                    ids=[item["id"] for item in batch],
                    metadata=[item["metadata"] for item in batch]
                )
                
            logging.info(f"Updated {len(embeddings)} products in vector index")
            
        except Exception as e:
            logging.error(f"Index update error: {str(e)}")
            raise
            
    async def refresh_index(self):
        """Refresh the index with latest product embeddings"""
        try:
            from src.inference.serving import ModelServer
            
            # Get model server instance
            model_server = ModelServer()
            
            # Get all products from BigQuery
            query = """
            SELECT *
            FROM `product_features_enriched`
            """
            
            # Generate new embeddings
            product_embeddings = {}
            product_metadata = {}
            
            async for row in self.bq_client.query(query).result():
                product_id = row["product_id"]
                
                # Generate embedding
                embedding = await model_server.get_product_embedding(dict(row))
                product_embeddings[product_id] = embedding
                
                # Store metadata
                product_metadata[product_id] = {
                    "category_id": row["category_id"],
                    "department_id": row["department_id"],
                    "last_modified_date": row["last_modified_date"].isoformat()
                }
            
            # Update index
            await self.update_index(product_embeddings, product_metadata)
            
            logging.info("Successfully refreshed vector index")
            
        except Exception as e:
            logging.error(f"Index refresh error: {str(e)}")
            raise
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = await self.index.get_stats_async()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_size_bytes": stats.index_size_bytes,
                "updated_at": stats.last_update_time.isoformat()
            }
        except Exception as e:
            logging.error(f"Error getting index stats: {str(e)}")
            raise