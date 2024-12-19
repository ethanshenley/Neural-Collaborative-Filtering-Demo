CREATE OR REPLACE VIEW user_interactions AS
SELECT 
    CONCAT(thf.store_number, '-', thf.transaction_number, '-', tbf.line_number) as interaction_id,
    thf.store_number,
    thf.cust_code as user_id,
    tbf.inventory_code as product_id,
    'purchase' as interaction_type,  -- Since these are all from transactions
    tbf.sold_quantity as quantity,
    tbf.extended_retail as amount,
    thf.physical_date_time as transaction_timestamp,
    thf.business_date,
    COALESCE(tbf.last_modified_date, thf.last_modified_date) as last_modified_date
FROM transaction_header_fact thf
JOIN transaction_body_fact tbf 
    ON thf.store_number = tbf.store_number 
    AND thf.transaction_number = tbf.transaction_number
WHERE thf.cust_code IS NOT NULL;  -- Only including loyalty customer transactions



CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    thf.cust_code as user_id,
    COUNT(DISTINCT CONCAT(thf.store_number, '-', thf.transaction_number)) as total_purchases,
    SUM(tbf.extended_retail) as total_amount,
    AVG(tbf.sold_quantity) as avg_basket_size,
    FIRST_VALUE(tbf.category_id) OVER (
        PARTITION BY thf.cust_code 
        ORDER BY COUNT(*) DESC
    ) as favorite_category,
    FIRST_VALUE(tbf.department_id) OVER (
        PARTITION BY thf.cust_code 
        ORDER BY COUNT(*) DESC
    ) as favorite_department,
    MAX(thf.physical_date_time) as last_purchase_timestamp,
    -- Calculate average days between purchases
    AVG(DATE_DIFF(
        physical_date_time, 
        LAG(physical_date_time) OVER (
            PARTITION BY thf.cust_code 
            ORDER BY physical_date_time
        ), 
        DAY
    )) as purchase_frequency_days,
    CURRENT_TIMESTAMP() as last_calculated_at
FROM transaction_header_fact thf
JOIN transaction_body_fact tbf 
    ON thf.store_number = tbf.store_number 
    AND thf.transaction_number = tbf.transaction_number
WHERE thf.cust_code IS NOT NULL
GROUP BY thf.cust_code;




CREATE OR REPLACE VIEW product_statistics AS
SELECT 
    tbf.inventory_code as product_id,
    COUNT(*) as total_sales,
    SUM(tbf.extended_retail) as total_revenue,
    COUNT(DISTINCT thf.cust_code) as unique_customers,
    AVG(tbf.sold_quantity) as avg_purchase_quantity,
    CURRENT_TIMESTAMP() as last_calculated_at
FROM transaction_body_fact tbf
JOIN transaction_header_fact thf 
    ON thf.store_number = tbf.store_number 
    AND thf.transaction_number = tbf.transaction_number
WHERE thf.cust_code IS NOT NULL  -- Only counting loyalty customer purchases
GROUP BY tbf.inventory_code;