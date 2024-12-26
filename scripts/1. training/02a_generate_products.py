import logging
from src.llm.vertex_client import VertexLLMClient
from src.llm.product_prompts import ProductPromptGenerator
from src.llm.response_parser import ProductResponseParser
from src.data.bigquery.data_loader import BigQueryLoader

def validate_category_code(dept_id: str, cat_id: str) -> bool:
    """Validate that category code matches our defined mappings"""
    # Get category type from category_id (e.g., 'SNK1' -> 'SNK')
    category_type = cat_id[:3]
    
    # Check if category type exists in our mapping
    if category_type not in ProductPromptGenerator.CATEGORY_CODES:
        return False
        
    # Check if specific category code is valid for this type
    valid_codes = ProductPromptGenerator.CATEGORY_CODES[category_type]
    return cat_id in valid_codes

def generate_and_load_products(products_per_subcategory: int = 40):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize clients
    vertex_client = VertexLLMClient(project_id="sheetz-poc")
    bq_loader = BigQueryLoader()
    
    CATEGORY_DISTRIBUTION = {
        ("FS", "MTO1"): 15, ("FS", "MTO2"): 15, ("FS", "MTO3"): 15,
        ("FS", "RTE1"): 12, ("FS", "RTE2"): 12,
        ("FS", "HOT1"): 10, ("FS", "HOT2"): 10,
        ("BAK", "BAK1"): 10, ("BAK", "BAK2"): 10,
        ("BV", "BEV1"): 15, ("BV", "BEV2"): 15, ("BV", "BEV3"): 15,
        ("BV", "COF1"): 10, ("BV", "COF2"): 10,
        ("BV", "FTN1"): 10, ("BV", "FTN2"): 10,
        ("SC", "SNK1"): 12, ("SC", "SNK2"): 12, ("SC", "SNK3"): 12,
        ("SC", "CND1"): 10, ("SC", "CND2"): 10,
        ("GR", "GRO1"): 8, ("GR", "GRO2"): 8,
        ("GR", "DRY1"): 8,
       # ("TA", "TOB1"): 10, ("TA", "TOB2"): 10,
      #  ("TA", "ALC1"): 12, ("TA", "ALC2"): 12
    }
    
    all_products = []
    max_retries = 3  # Limit retries per batch

    for (dept_id, cat_id), target_count in CATEGORY_DISTRIBUTION.items():
        if not validate_category_code(dept_id, cat_id):
            logging.error(f"Invalid category code: {dept_id}/{cat_id}")
            continue

        logging.info(f"\nGenerating {target_count} products for {dept_id}/{cat_id}")
        remaining = target_count
        
        while remaining > 0:
            batch_size = min(5, remaining)
            prompt = ProductPromptGenerator.generate_prompt(
                department_id=dept_id,
                category_id=cat_id,
                count=batch_size
            )

            success = False
            for attempt in range(max_retries):
                response = vertex_client.generate(prompt)
                if response:
                    # Parse and validate products
                    products = []
                    for line in response.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                product = ProductResponseParser.parse_product_line(line, dept_id)
                                if product and product['category_id'] == cat_id:
                                    products.append(product)
                                else:
                                    logging.warning(f"Invalid product category: {line}")
                            except Exception as e:
                                logging.error(f"Error parsing line: {line}\nError: {str(e)}")

                    if products:
                        all_products.extend(products)
                        remaining -= len(products)
                        if len(products) < batch_size:
                            logging.warning(f"Only generated {len(products)} of {batch_size} requested products")
                        success = True
                        break
                    else:
                        logging.warning(f"No valid products generated on attempt {attempt + 1} for {dept_id}/{cat_id}")
                else:
                    logging.warning(f"No response on attempt {attempt + 1} for {dept_id}/{cat_id}")

            if not success:
                # Could not generate products for this category after max_retries
                logging.error(f"Failed to generate products for {dept_id}/{cat_id} after {max_retries} retries.")
                # Break out of this category to avoid infinite loop
                break
    
    # Load to BigQuery
    if all_products:
        if bq_loader.clear_table("product_features"):
            if bq_loader.load_products(all_products):
                logging.info(f"Successfully loaded {len(all_products)} total products")
            else:
                logging.error("Failed to load products to BigQuery")
    else:
        logging.error("No products generated!")

if __name__ == "__main__":
    generate_and_load_products()