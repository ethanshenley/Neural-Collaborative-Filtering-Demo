from google.cloud import bigquery
import logging

class FeatureViewCreator:
    def __init__(self, project_id: str = "sheetz-poc", dataset_id: str = "sheetz_data"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_user_features_view(self) -> None:
        query = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.user_features_enriched` AS
        WITH user_sequences AS (
          SELECT 
            thf.cust_code as user_id,
            ARRAY_AGG(STRUCT(
              tbf.inventory_code as product_id,
              tbf.department_id,
              tbf.category_id,
              thf.physical_date_time as transaction_timestamp,
              tbf.extended_retail as amount
            ) ORDER BY thf.physical_date_time DESC LIMIT 50) as recent_interactions,
            MIN(thf.physical_date_time) as first_interaction,
            MAX(thf.physical_date_time) as last_interaction,
            COUNT(*) as total_interactions
          FROM `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
          JOIN `{self.project_id}.{self.dataset_id}.transaction_body_fact` tbf 
            ON thf.store_number = tbf.store_number 
            AND thf.transaction_number = tbf.transaction_number
          WHERE thf.cust_code IS NOT NULL
          GROUP BY thf.cust_code
        ),

        category_counts AS (
          SELECT 
            thf.cust_code as user_id,
            tbf.department_id,
            tbf.category_id,
            COUNT(*) as purchase_count,
            SUM(tbf.extended_retail) as total_spend
          FROM `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
          JOIN `{self.project_id}.{self.dataset_id}.transaction_body_fact` tbf 
            ON thf.store_number = tbf.store_number 
            AND thf.transaction_number = tbf.transaction_number
          WHERE thf.cust_code IS NOT NULL
          GROUP BY thf.cust_code, tbf.department_id, tbf.category_id
        ),

        category_preferences AS (
          SELECT
            user_id,
            ARRAY_AGG(
              STRUCT(
                department_id,
                category_id,
                purchase_count,
                total_spend
              )
              ORDER BY purchase_count DESC
              LIMIT 5
            ) as preferred_categories
          FROM category_counts
          GROUP BY user_id
        ),

        hourly_counts AS (
          SELECT
            thf.cust_code as user_id,
            EXTRACT(HOUR FROM thf.physical_date_time) as hour,
            COUNT(*) as visit_count
          FROM `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
          WHERE thf.cust_code IS NOT NULL
          GROUP BY thf.cust_code, hour
        ),

        daily_counts AS (
          SELECT
            thf.cust_code as user_id,
            EXTRACT(DAYOFWEEK FROM thf.physical_date_time) as day,
            COUNT(*) as visit_count
          FROM `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
          WHERE thf.cust_code IS NOT NULL
          GROUP BY thf.cust_code, day
        ),

        temporal_patterns AS (
          SELECT 
            h.user_id,
            ARRAY_AGG(
              STRUCT(
                h.hour,
                h.visit_count
              ) ORDER BY h.visit_count DESC
            ) as hourly_pattern,
            ARRAY_AGG(
              STRUCT(
                d.day,
                d.visit_count
              ) ORDER BY d.visit_count DESC
            ) as daily_pattern
          FROM hourly_counts h
          JOIN daily_counts d
            ON h.user_id = d.user_id
          GROUP BY h.user_id
        )

        SELECT
          lcd.*,
          us.recent_interactions,
          us.first_interaction,
          us.last_interaction,
          us.total_interactions,
          cp.preferred_categories,
          tp.hourly_pattern,
          tp.daily_pattern,
          
          -- Derived features
          DATE_DIFF(CURRENT_DATE(), lcd.activation_date, DAY) as account_age_days,
          SAFE_DIVIDE(us.total_interactions, 
            NULLIF(DATE_DIFF(CURRENT_DATE(), lcd.activation_date, DAY), 0)) as interaction_frequency,
          SAFE_DIVIDE(lcd.lifetime_points, 
            NULLIF(us.total_interactions, 0)) as points_per_interaction

        FROM `{self.project_id}.{self.dataset_id}.loyalty_customer_dim` lcd
        LEFT JOIN user_sequences us
          ON lcd.cardnumber = us.user_id
        LEFT JOIN category_preferences cp
          ON lcd.cardnumber = cp.user_id
        LEFT JOIN temporal_patterns tp
          ON lcd.cardnumber = tp.user_id
        WHERE lcd.enrollment_status = 1
        """
        
        try:
            self.client.query(query).result()
            self.logger.info("Successfully created user_features_enriched view")
        except Exception as e:
            self.logger.error(f"Error creating user features view: {str(e)}")
            raise

    def create_product_features_view(self) -> None:
        query = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.product_features_enriched` AS 
        WITH product_stats AS (
          SELECT
            tbf.inventory_code as product_id,
            COUNT(DISTINCT thf.cust_code) as unique_customers,
            COUNT(*) as total_purchases,
            SUM(tbf.extended_retail) as total_revenue,
            AVG(tbf.extended_retail) as avg_price,
            AVG(tbf.sold_quantity) as avg_quantity
          FROM `{self.project_id}.{self.dataset_id}.transaction_body_fact` tbf
          JOIN `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
            ON tbf.store_number = thf.store_number 
            AND tbf.transaction_number = thf.transaction_number
          WHERE thf.cust_code IS NOT NULL
          GROUP BY tbf.inventory_code
        ),

        pair_counts AS (
          SELECT 
            t1.inventory_code as product_id,
            t2.inventory_code as paired_product_id,
            COUNT(*) as pair_count,
            COUNT(DISTINCT t1.store_number || '-' || t1.transaction_number) as total_transactions
          FROM `{self.project_id}.{self.dataset_id}.transaction_body_fact` t1
          JOIN `{self.project_id}.{self.dataset_id}.transaction_body_fact` t2
            ON t1.store_number = t2.store_number
            AND t1.transaction_number = t2.transaction_number
            AND t1.inventory_code != t2.inventory_code
          GROUP BY t1.inventory_code, t2.inventory_code
        ),

        product_pairs AS (
          SELECT 
            product_id,
            ARRAY_AGG(
              STRUCT(
                paired_product_id,
                pair_count,
                SAFE_DIVIDE(pair_count, total_transactions) as pair_ratio
              ) 
              ORDER BY pair_count DESC
              LIMIT 10
            ) as common_pairs
          FROM pair_counts
          GROUP BY product_id
        ),

        hourly_sales_counts AS (
          SELECT
            tbf.inventory_code as product_id,
            EXTRACT(HOUR FROM thf.physical_date_time) as hour,
            COUNT(*) as purchase_count
          FROM `{self.project_id}.{self.dataset_id}.transaction_body_fact` tbf
          JOIN `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
            ON tbf.store_number = thf.store_number 
            AND tbf.transaction_number = thf.transaction_number
          GROUP BY tbf.inventory_code, hour
        ),

        daily_sales_counts AS (
          SELECT
            tbf.inventory_code as product_id,
            EXTRACT(DAYOFWEEK FROM thf.physical_date_time) as day,
            COUNT(*) as purchase_count
          FROM `{self.project_id}.{self.dataset_id}.transaction_body_fact` tbf
          JOIN `{self.project_id}.{self.dataset_id}.transaction_header_fact` thf
            ON tbf.store_number = thf.store_number 
            AND tbf.transaction_number = thf.transaction_number
          GROUP BY tbf.inventory_code, day
        ),

        time_patterns AS (
          SELECT
            h.product_id,
            ARRAY_AGG(
              STRUCT(
                h.hour,
                h.purchase_count
              ) ORDER BY h.purchase_count DESC
            ) as hourly_sales,
            ARRAY_AGG(
              STRUCT(
                d.day,
                d.purchase_count
              ) ORDER BY d.purchase_count DESC
            ) as daily_sales
          FROM hourly_sales_counts h
          JOIN daily_sales_counts d
            ON h.product_id = d.product_id
          GROUP BY h.product_id
        )

        SELECT
          pf.*,
          ps.unique_customers,
          ps.total_purchases,
          ps.total_revenue,
          ps.avg_price,
          ps.avg_quantity,
          pp.common_pairs,
          tp.hourly_sales,
          tp.daily_sales,
          SAFE_DIVIDE(ps.total_revenue, ps.total_purchases) as revenue_per_purchase,
          SAFE_DIVIDE(ps.unique_customers, ps.total_purchases) as purchase_loyalty_score

        FROM `{self.project_id}.{self.dataset_id}.product_features` pf
        LEFT JOIN product_stats ps
          ON pf.product_id = ps.product_id  
        LEFT JOIN product_pairs pp
          ON pf.product_id = pp.product_id
        LEFT JOIN time_patterns tp
          ON pf.product_id = tp.product_id
        """
        
        try:
            self.client.query(query).result()
            self.logger.info("Successfully created product_features_enriched view")
        except Exception as e:
            self.logger.error(f"Error creating product features view: {str(e)}")
            raise

    def create_all_views(self) -> None:
        self.logger.info("Creating enriched feature views...")
        self.create_user_features_view()
        self.create_product_features_view()
        self.logger.info("Successfully created all feature views")

def main():
    creator = FeatureViewCreator()
    creator.create_all_views()

if __name__ == "__main__":
    main()