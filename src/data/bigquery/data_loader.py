from google.cloud import bigquery
from typing import List, Dict, Any
import logging
from datetime import datetime, date

class BigQueryLoader:
    def __init__(self, project_id: str = "sheetz-poc", dataset_id: str = "sheetz_data"):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.project_id = project_id
        
    def load_products(self, products: List[Dict[str, Any]]) -> bool:
        """Load products into BigQuery product_features table"""
        table_id = f"{self.project_id}.{self.dataset_id}.product_features"
        
        try:
            # Get table reference
            table = self.client.get_table(table_id)
            
            # Ensure datetime is in the correct format
            formatted_products = []
            for product in products:
                formatted_product = product.copy()
                if isinstance(product['last_modified_date'], datetime):
                    formatted_product['last_modified_date'] = product['last_modified_date'].strftime('%Y-%m-%d %H:%M:%S')
                formatted_products.append(formatted_product)
            
            # Insert rows
            errors = self.client.insert_rows_json(table, formatted_products)
            
            if errors:
                logging.error(f"Encountered errors while inserting rows: {errors}")
                return False
                
            logging.info(f"Successfully loaded {len(products)} products into BigQuery")
            return True
            
        except Exception as e:
            logging.error(f"Error loading products to BigQuery: {str(e)}")
            return False
   
    def load_customers(self, customers: List[Dict[str, Any]]) -> bool:
        """Load customers into BigQuery customer_loyalty_dim table"""
        table_id = f"{self.project_id}.{self.dataset_id}.loyalty_customer_dim"
        
        try:
            # Get table reference
            table = self.client.get_table(table_id)
            
            # Ensure datetime fields are properly formatted
            formatted_customers = []
            for customer in customers:
                formatted_customer = customer.copy()
                # Convert datetime objects to string format
                for field in ['created_at', 'updated_at', 'load_date']:
                    if isinstance(customer[field], datetime):
                        formatted_customer[field] = customer[field].strftime('%Y-%m-%d %H:%M:%S')
                # Convert date objects to string format
                for field in ['activation_date', 'cancellation_date', 'last_purchase_date', 
                            'last_redemption_date', 'birth_date']:
                    if customer[field] and isinstance(customer[field], date):
                        formatted_customer[field] = customer[field].strftime('%Y-%m-%d')
                formatted_customers.append(formatted_customer)
            
            # Insert rows
            errors = self.client.insert_rows_json(table, formatted_customers)
            
            if errors:
                logging.error(f"Encountered errors while inserting rows: {errors}")
                return False
                
            logging.info(f"Successfully loaded {len(customers)} customers into BigQuery")
            return True
            
        except Exception as e:
            logging.error(f"Error loading customers to BigQuery: {str(e)}")
            return False
        
    def load_transaction_headers(self, headers: List[Dict], batch_size: int = 1000) -> bool:
        """Load transaction headers to BigQuery in batches"""
        table_id = f"{self.project_id}.{self.dataset_id}.transaction_header_fact"
        
        try:
            table = self.client.get_table(table_id)
            total_loaded = 0
            
            for i in range(0, len(headers), batch_size):
                batch = headers[i:i + batch_size]
                
                # Format dates for JSON serialization
                formatted_batch = []
                for header in batch:
                    formatted_header = {}
                    for key, value in header.items():
                        if isinstance(value, (datetime, date)):
                            formatted_header[key] = value.isoformat()
                        else:
                            formatted_header[key] = value
                    formatted_batch.append(formatted_header)
                
                errors = self.client.insert_rows_json(table, formatted_batch)
                if errors:
                    logging.error(f"Errors loading headers batch {i//batch_size}: {errors}")
                    return False
                
                total_loaded += len(batch)
                logging.info(f"Loaded {total_loaded}/{len(headers)} headers")
                
            return True
        except Exception as e:
            logging.error(f"Error loading headers: {str(e)}")
            return False

    def load_transaction_items(self, items: List[Dict], batch_size: int = 1000) -> bool:
        """Load transaction items to BigQuery in batches"""
        table_id = f"{self.project_id}.{self.dataset_id}.transaction_body_fact"
        
        try:
            table = self.client.get_table(table_id)
            total_loaded = 0
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Format dates for JSON serialization
                formatted_batch = []
                for item in batch:
                    formatted_item = {}
                    for key, value in item.items():
                        if isinstance(value, (datetime, date)):
                            formatted_item[key] = value.isoformat()
                        else:
                            formatted_item[key] = value
                    formatted_batch.append(formatted_item)
                
                errors = self.client.insert_rows_json(table, formatted_batch)
                if errors:
                    logging.error(f"Errors loading items batch {i//batch_size}: {errors}")
                    return False
                
                total_loaded += len(batch)
                logging.info(f"Loaded {total_loaded}/{len(items)} items")
                
            return True
        except Exception as e:
            logging.error(f"Error loading items: {str(e)}")
            return False


    def clear_table(self, table_name: str) -> bool:
        """Clear all rows from a table by truncating it."""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        try:
            query = f"TRUNCATE TABLE `{table_id}`"
            self.client.query(query).result()  # Wait for completion
            logging.info(f"Successfully truncated table {table_name}")
            return True
        except Exception as e:
            logging.error(f"Error truncating table {table_name}: {str(e)}")
            return False
                    
    def execute_query(self, query: str) -> List[bigquery.Row]:
        """Execute a BigQuery query and return the results."""
        try:
            query_job = self.client.query(query)
            results = list(query_job.result())  # Fetch query results
            logging.info(f"Successfully executed query: {query}")
            return results
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            return []
    
    def _recreate_table_with_partitioning(self, table_name: str) -> bool:
        """Recreate table with proper partitioning"""
        try:
            # First drop the existing table
            drop_query = f"""
            DROP TABLE IF EXISTS `{self.project_id}.{self.dataset_id}.{table_name}`
            """
            self.client.query(drop_query).result()  # Wait for completion
            logging.info(f"Dropped table {table_name}")

            # Then recreate with proper partitioning
            if table_name == "transaction_header_fact":
                create_query = f"""
                CREATE TABLE `{self.project_id}.{self.dataset_id}.{table_name}`
                (
                    store_number INT64,
                    close_date DATE,
                    transaction_number INT64,
                    business_date DATE,
                    physical_date DATE,
                    physical_date_time TIMESTAMP,
                    physical_time STRING,
                    printed STRING,
                    employee_code STRING,
                    discount FLOAT64,
                    tax_total FLOAT64,
                    method_of_payment_code1 STRING,
                    method_of_payment_amt1 FLOAT64,
                    drawer_number INT64,
                    method_of_payment_code2 STRING,
                    method_of_payment_amt2 FLOAT64,
                    refund FLOAT64,
                    change FLOAT64,
                    void_code STRING,
                    taxable STRING,
                    shift_number INT64,
                    bill_status STRING,
                    station_number INT64,
                    sub_total FLOAT64,
                    cust_code STRING,
                    cc_code STRING,
                    ar_ref_number STRING,
                    tax_1 FLOAT64,
                    tax_2 FLOAT64,
                    print_count INT64,
                    food_change FLOAT64,
                    method_of_payment_code3 STRING,
                    method_of_payment_amt3 FLOAT64,
                    method_of_payment_code4 STRING,
                    method_of_payment_amt4 FLOAT64,
                    method_of_payment_code5 STRING,
                    method_of_payment_amt5 FLOAT64,
                    method_of_payment_code6 STRING,
                    method_of_payment_amt6 FLOAT64,
                    method_of_payment_code7 STRING,
                    method_of_payment_amt7 FLOAT64,
                    method_of_payment_code8 STRING,
                    method_of_payment_amt8 FLOAT64,
                    method_of_payment_code9 STRING,
                    method_of_payment_amt9 FLOAT64,
                    transaction_source INT64,
                    event_id INT64,
                    reason_name STRING,
                    tax_3 FLOAT64,
                    tax_4 FLOAT64,
                    tax_5 FLOAT64,
                    tax_6 FLOAT64,
                    tax_7 FLOAT64,
                    tax_8 FLOAT64,
                    tax_9 FLOAT64,
                    tax_10 FLOAT64,
                    tax_11 FLOAT64,
                    tax_12 FLOAT64,
                    tax_13 FLOAT64,
                    tax_14 FLOAT64,
                    tax_15 FLOAT64,
                    last_modified_date TIMESTAMP
                )
                PARTITION BY DATE(physical_date_time)
                CLUSTER BY store_number, transaction_number;
                """
            elif table_name == "transaction_body_fact":
                create_query = f"""
                CREATE TABLE `{self.project_id}.{self.dataset_id}.{table_name}`
                (
                    store_number INT64,
                    close_date DATE,
                    business_date DATE,
                    physical_date STRING,
                    transaction_number INT64,
                    inventory_code STRING,
                    sold_quantity FLOAT64,
                    extended_cost FLOAT64,
                    adjusted STRING,
                    description STRING,
                    inventory_type STRING,
                    department_category_description STRING,
                    extended_retail FLOAT64,
                    tax_class STRING,
                    unit_retail FLOAT64,
                    unit_size_code STRING,
                    unit_size_quantity INT64,
                    unit_cost FLOAT64,
                    barcode STRING,
                    shift_number INT64,
                    drawer_number INT64,
                    special_discount STRING,
                    bill_status STRING,
                    scanned STRING,
                    extended_retail_adjustment FLOAT64,
                    status STRING,
                    fuel_volume FLOAT64,
                    prepaid FLOAT64,
                    pump_number INT64,
                    fuel_amount FLOAT64,
                    unit_retail_org FLOAT64,
                    foodstamp STRING,
                    prepay_transaction STRING,
                    food_amount FLOAT64,
                    item_code_ep STRING,
                    department_category_ep STRING,
                    department_id STRING,
                    category_id STRING,
                    item_number INT64,
                    parent_item_number INT64,
                    detail_cancel_flag_value STRING,
                    detail_cancel_flag STRING,
                    last_modified_date TIMESTAMP,
                    line_number INT64
                )
                PARTITION BY business_date
                CLUSTER BY store_number, transaction_number;
                """
            
            # Execute the creation query and wait for completion
            job = self.client.query(create_query)
            job.result()  # This will wait for the query to complete

            # Verify table exists using get_table
            try:
                table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{table_name}")
                logging.info(f"Successfully recreated table {table_name} with {len(table.schema)} columns")
                return True
            except Exception as e:
                logging.error(f"Table creation verification failed for {table_name}: {str(e)}")
                return False
                    
        except Exception as e:
            logging.error(f"Error recreating table {table_name}: {str(e)}")
            return False