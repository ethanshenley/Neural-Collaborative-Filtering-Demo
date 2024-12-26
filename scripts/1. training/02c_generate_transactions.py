# scripts/02c_generate_transactions.py

import logging
from datetime import datetime, timedelta
from src.data.generators.transaction_generator import TransactionGenerator
from src.data.bigquery.data_loader import BigQueryLoader

def generate_and_load_transactions(start_date: datetime, num_days: int, transactions_per_day: int):
    """Generate and load transactions to BigQuery"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    bq_loader = BigQueryLoader()
    transaction_gen = TransactionGenerator(bq_loader)
    
    end_date = start_date + timedelta(days=num_days - 1)
    logging.info(f"Generating transactions from {start_date.date()} to {end_date.date()}")
    
    try:
        # Truncate existing tables instead of recreating them
        if bq_loader.clear_table("transaction_header_fact"):
            logging.info("Cleared transaction_header_fact table")
        else:
            logging.error("Failed to clear transaction_header_fact table")
            return  # Exit if unable to clear table

        if bq_loader.clear_table("transaction_body_fact"):
            logging.info("Cleared transaction_body_fact table")
        else:
            logging.error("Failed to clear transaction_body_fact table")
            return  # Exit if unable to clear table
        
        # Generate transactions
        headers, items = transaction_gen.generate_batch(
            start_date=start_date,
            end_date=end_date,
            transactions_per_day=transactions_per_day
        )
        
        # Load to BigQuery
        if bq_loader.load_transaction_headers(headers):
            logging.info(f"Loaded {len(headers)} transaction headers")
        else:
            logging.error("Failed to load transaction headers")
        
        if bq_loader.load_transaction_items(items):
            logging.info(f"Loaded {len(items)} transaction items")
        else:
            logging.error("Failed to load transaction items")
        
        # Print summary statistics
        logging.info("\nTransaction Summary:")
        results = bq_loader.execute_query("""
            SELECT 
                DATE(th.business_date) as transaction_date,
                COUNT(DISTINCT th.transaction_number) as num_transactions,
                COUNT(DISTINCT th.cust_code) as unique_customers,
                SUM(th.sub_total) as total_sales
            FROM `sheetz-poc.sheetz_data.transaction_header_fact` th
            GROUP BY DATE(th.business_date)
            ORDER BY transaction_date
        """)
        
        for row in results:
            logging.info(
                f"Date: {row.transaction_date}, Transactions: {row.num_transactions}, "
                f"Customers: {row.unique_customers}, Sales: ${row.total_sales:,.2f}"
            )
    except Exception as e:
        logging.error(f"Error in transaction generation: {str(e)}")
        raise

if __name__ == "__main__":
    start_date = datetime(2024, 1, 1)  # Start from January 1st, 2024
    num_days = 90  # Generate 90 days of transactions
    transactions_per_day = 1000  # 1000 transactions per day
    
    generate_and_load_transactions(start_date, num_days, transactions_per_day)
