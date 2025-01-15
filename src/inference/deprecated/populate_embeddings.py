# populate_embeddings.py
import asyncio
import logging
import torch
import numpy as np

from google.cloud import aiplatform
from google.cloud import bigquery
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from src.inference.serving import ModelServer  # your existing import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "726149969503"
REGION = "us-central1"

# Index Endpoint and Index resource name
INDEX_ENDPOINT_ID = "8020519510207365120"
INDEX_RESOURCE_NAME = (
    "projects/726149969503/locations/us-central1/indexes/3832382962985336832"
)

# BQ table
PRODUCT_FEATURES_TABLE = "sheetz-poc.sheetz_data.product_features_enriched"


async def populate_vector_search():
    """Populate vector search with product embeddings from trained model."""
    try:
        # 1) Initialize clients
        logger.info("Initializing clients...")
        model_server = ModelServer()
        bq_client = bigquery.Client(project=PROJECT_ID)

        # 2) Query your product_features_enriched table
        query = f"""
        SELECT
            product_id,
            category_id,
            department_id,
            product_name,
            unique_customers,
            total_purchases,
            total_revenue,
            avg_price,
            avg_quantity,
            purchase_loyalty_score,
            hourly_sales,
            daily_sales
        FROM `{PRODUCT_FEATURES_TABLE}`
        """
        logger.info("Querying BigQuery...")
        df = bq_client.query(query).to_dataframe()
        logger.info(f"Found {len(df)} products")

        if df.empty:
            logger.warning("No products found! Exiting early.")
            return

        # Show sample row structure
        logger.info("\nSample row structure:")
        sample_row = df.iloc[0]
        for col in df.columns:
            logger.info(f"{col}: {type(sample_row[col]).__name__}")
            if col in ["hourly_sales", "daily_sales"]:
                logger.info(f"{col} sample: {sample_row[col][:2]}")

        # 3) Build ID mappings
        unique_product_ids = df["product_id"].unique()
        unique_category_ids = df["category_id"].unique()
        unique_department_ids = df["department_id"].unique()

        product_id_to_idx = {pid: i for i, pid in enumerate(unique_product_ids)}
        category_id_to_idx = {cid: i for i, cid in enumerate(unique_category_ids)}
        department_id_to_idx = {did: i for i, did in enumerate(unique_department_ids)}

        logger.info(f"\nCreated ID mappings:")
        logger.info(f"  Products: {len(product_id_to_idx)}")
        logger.info(f"  Categories: {len(category_id_to_idx)}")
        logger.info(f"  Departments: {len(department_id_to_idx)}")

        # 4) Generate datapoints
        datapoints = []
        for _, row in df.iterrows():
            try:
                logger.info(f"\nProcessing product {row['product_id']}")
                product_idx = product_id_to_idx[row["product_id"]]

                # Build features for get_product_embeddings
                keys = ["user_id", "product_id"]
                values = torch.tensor([0, product_idx], dtype=torch.long)  # user=0 (dummy)
                lengths = torch.tensor([1, 1], dtype=torch.long)
                offsets = torch.tensor([0, 1, 2], dtype=torch.long)

                features = {
                    "product_features": KeyedJaggedTensor(
                        keys=keys,
                        values=values,
                        lengths=lengths,
                        offsets=offsets
                    ),
                    "category_features": {
                        "department_ids": torch.tensor(
                            [department_id_to_idx[row["department_id"]]], dtype=torch.long
                        ),
                        "category_ids": torch.tensor(
                            [category_id_to_idx[row["category_id"]]], dtype=torch.long
                        ),
                    },
                }

                # Log
                logger.info("Input feature structure:")
                logger.info(f"  Keys: {keys}")
                logger.info(f"  Values: {values}")
                logger.info(f"  Lengths: {lengths}")
                logger.info(f"  Offsets: {offsets}")

                # Generate embeddings
                with torch.no_grad():
                    emb_dict = model_server.model.get_product_embeddings(features)
                    logger.info("Generated embeddings:")
                    for k, v in emb_dict.items():
                        logger.info(f"  - {k} shape: {v.shape}")

                    # Example: take "mlp"
                    final_embedding_tensor = emb_dict["mlp"][0]
                    final_embedding = final_embedding_tensor.cpu().numpy().tolist()

                # Convert your metadata into "restrictions" if you want to filter on them
                # "restrictions" is a list of objects with "namespace" and "allow"
                # e.g. restricting on department_id & category_id
                restrictions = [
                    {
                        "namespace": "department_id",
                        "allow": [str(row["department_id"])],
                    },
                    {
                        "namespace": "category_id",
                        "allow": [str(row["category_id"])],
                    },
                ]
                # You can add more as needed

                # Build the final datapoint
                datapoint = {
                    # REQUIRED: must be "datapoint_id", not "id"
                    "datapoint_id": str(row["product_id"]),  
                    # REQUIRED: must be "feature_vector", not "embedding"
                    "feature_vector": final_embedding,
                    # OPTIONAL: "restrictions"
                }

                datapoints.append(datapoint)
                logger.info(f"Successfully processed product {row['product_id']}")

            except Exception as e:
                logger.error(f"Error processing product {row['product_id']}: {e}")
                raise

        logger.info(f"\nReady to upsert {len(datapoints)} embeddings to Vector Search...")

        # 5) Upsert datapoints
        endpoint_resource_name = f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_resource_name)

        matching_engine_index = aiplatform.MatchingEngineIndex(index_name=INDEX_RESOURCE_NAME)

        batch_size = 50
        for i in range(0, len(datapoints), batch_size):
            batch = datapoints[i : i + batch_size]
            try:
                matching_engine_index.upsert_datapoints(datapoints=batch)
                logger.info(
                    f"Uploaded batch {(i // batch_size) + 1} / "
                    f"{(len(datapoints) + batch_size - 1) // batch_size}"
                )
            except Exception as e:
                logger.error(f"Error uploading batch {(i // batch_size) + 1}: {str(e)}")
                continue

        logger.info(f"Successfully populated vector search with {len(datapoints)} products")

    except Exception as e:
        logger.error(f"Error in populate_vector_search: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(populate_vector_search())
