import uuid
import yaml
import logging
from google.cloud import bigquery

def export_bq_view_to_gcs(config_path: str = "config/config.yaml"):
    """Materialize a view into a table, then export that table to GCS as newline-delimited JSON."""
    # 1) Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_id = config["gcp"]["project_id"]
    dataset_id = config["gcp"]["dataset_id"]
    staging_bucket = config["gcp"]["staging_bucket"]

    bq_client = bigquery.Client(project=project_id)

    # 2) Create a temp table from the view
    temp_table_id = f"{project_id}.{dataset_id}.temp_export_{uuid.uuid4().hex}"

    query = f"""
    CREATE OR REPLACE TABLE `{temp_table_id}` AS
    SELECT *
    FROM `{project_id}.{dataset_id}.product_features_enriched`  -- This is a VIEW
    """
    logging.info(f"Creating temporary table from the view: {temp_table_id}")
    bq_client.query(query).result()

    # 3) Export that table to GCS
    destination_uri = f"gs://{staging_bucket}/exports/product_data_{uuid.uuid4()}/*.json"
    logging.info(f"Exporting {temp_table_id} to {destination_uri}...")

    job_config = bigquery.job.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
    )
    extract_job = bq_client.extract_table(temp_table_id, destination_uri, job_config=job_config)
    extract_job.result()
    logging.info("Export job completed successfully.")

    # 4) Optional: drop the temp table
    bq_client.delete_table(temp_table_id)
    logging.info(f"Dropped temp table {temp_table_id}. Done!")

if __name__ == "__main__":
    export_bq_view_to_gcs()
