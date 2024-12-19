# scripts/02b_generate_customers.py

import logging
from src.data.generators.loyalty_customer_generator import LoyaltyCustomerGenerator
from src.data.bigquery.data_loader import BigQueryLoader
from random import random
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

def generate_and_load_customers(num_customers: int = 10000):
    """Generate customers and load to BigQuery"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    customer_gen = LoyaltyCustomerGenerator(seed=42)
    bq_loader = BigQueryLoader()
    
    logging.info(f"Generating {num_customers} customers...")
    
    BATCH_SIZE = 500
    total_loaded = 0

    # Delete and recreate table with proper partitioning
    recreate_query = f"""
    CREATE OR REPLACE TABLE `{bq_loader.project_id}.{bq_loader.dataset_id}.loyalty_customer_dim`
    PARTITION BY DATE(load_date)
    CLUSTER BY cardnumber
    AS SELECT * FROM `{bq_loader.project_id}.{bq_loader.dataset_id}.loyalty_customer_dim` 
    WHERE 1=0
    """
    
    try:
        bq_loader.execute_query(recreate_query)
        logging.info("Successfully recreated table with proper partitioning")
    except Exception as e:
        logging.error(f"Error recreating table: {str(e)}")
        return
    
    # Generate and load in batches
    total_batches = (num_customers + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_num in range(total_batches):
        try:
            batch_size = min(BATCH_SIZE, num_customers - (batch_num * BATCH_SIZE))
            customers = customer_gen.generate_batch(batch_size)
            
            if bq_loader.load_customers(customers):
                total_loaded += len(customers)
                logging.info(f"Batch {batch_num + 1}/{total_batches}: Loaded {len(customers)} customers. Total: {total_loaded}")
            else:
                logging.error(f"Failed to load batch {batch_num + 1}")
                
            # Small delay between batches
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error in batch {batch_num + 1}: {str(e)}")
            continue

    # Verify final results
    logging.info("\nFinal customer distribution:")
    results = bq_loader.execute_query("""
        SELECT 
            state,
            enrollment_status,
            COUNT(*) as customer_count,
            AVG(age) as avg_age,
            AVG(lifetime_points) as avg_points,
            COUNT(CASE WHEN opt_in = 'Y' THEN 1 END) as opt_in_count,
            COUNT(CASE WHEN gender = 1 THEN 1 END) as female_count,
            COUNT(CASE WHEN gender = 2 THEN 1 END) as male_count
        FROM `sheetz-poc.sheetz_data.loyalty_customer_dim`
        GROUP BY state, enrollment_status
        ORDER BY state, enrollment_status
    """)
    
    for row in results:
        logging.info(
            f"State: {row.state}, Status: {row.enrollment_status}, "
            f"Count: {row.customer_count}, Avg Age: {row.avg_age:.1f}, "
            f"Opt-in Rate: {(row.opt_in_count/row.customer_count)*100:.1f}%, "
            f"Gender Split F/M: {row.female_count}/{row.male_count}"
        )

if __name__ == "__main__":
    generate_and_load_customers(10000)