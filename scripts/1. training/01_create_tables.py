# scripts/01_create_tables.py

import logging
from google.cloud import bigquery
from google.api_core import retry
from src.data.schemas import SCHEMA_DEFINITIONS, TABLE_CLUSTERING, TABLE_PARTITIONING  # Fixed import

# Constants
PROJECT_ID = "sheetz-poc"
PROJECT_NUMBER = "726149969503"
DATASET_ID = "sheetz_data"
LOCATION = "US"

def verify_project_access():
    client = bigquery.Client(project=PROJECT_ID)
    try:
        # Try getting the dataset if it exists
        try:
            client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
            logging.info(f"Found existing dataset {DATASET_ID}")
        except Exception:
            logging.info(f"Dataset {DATASET_ID} does not exist yet")
        
        # Verify we can list datasets (checks permissions)
        datasets = list(client.list_datasets())
        if datasets:
            logging.info("Successfully verified BigQuery access (datasets found).")
        else:
            logging.info("Successfully verified BigQuery access (no datasets found yet).")
        
        return client
    except Exception as e:
        logging.error(f"Failed to verify project access: {str(e)}")
        raise

def create_dataset(client: bigquery.Client) -> None:
    """Create BigQuery dataset if it doesn't exist"""
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    
    try:
        client.get_dataset(dataset_ref)
        logging.info(f"Dataset {dataset_ref} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        dataset = client.create_dataset(dataset, exists_ok=True)
        logging.info(f"Created dataset {dataset_ref}")

def create_table(client: bigquery.Client, 
                table_name: str,
                schema: list,
                clustering_fields: list = None,
                partition_field: str = None) -> None:
    """Create BigQuery table with specified schema and options"""
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    
    try:
        # Create TableReference object
        table = bigquery.Table(table_id, schema=schema)
        
        # Add clustering if specified
        if clustering_fields:
            table.clustering_fields = clustering_fields
            
        # Add partitioning if specified
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field
            )
            
        # Create table
        table = client.create_table(table, exists_ok=True)
        logging.info(f"Created table {table_id}")
        
        # Verify table was created
        client.get_table(table)
        logging.info(f"Verified table {table_id} exists and is accessible")
        
    except Exception as e:
        logging.error(f"Error creating table {table_id}: {str(e)}")
        raise

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Verify project access
    logging.info(f"Verifying access to project {PROJECT_ID}")
    client = verify_project_access()
    
    # Create dataset
    create_dataset(client)
    
    # Create tables from schemas
    for table_name, schema in SCHEMA_DEFINITIONS.items():
        logging.info(f"Creating table {table_name}")
        create_table(
            client=client,
            table_name=table_name,
            schema=schema,
            clustering_fields=TABLE_CLUSTERING.get(table_name),
            partition_field=(
                TABLE_PARTITIONING[table_name].field
                if table_name in TABLE_PARTITIONING
                else None
            )
        )
        logging.info(f"Successfully created table {table_name}")

if __name__ == "__main__":
    main()