import logging
from src.llm.vertex_client import VertexLLMClient
from src.llm.product_prompts import ProductPromptGenerator
from src.llm.response_parser import ProductResponseParser
import re

def test_product_generation():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize clients
    client = VertexLLMClient(project_id="ce-demo-space")
    
    # Test categories
    categories = ["MTO", "BEV", "HOT", "RTE"]
    
    for category in categories:
        logging.info(f"\nTesting {category} category:")
        
        # Generate prompt
        prompt = ProductPromptGenerator.generate_prompt(category, count=3)
        
        # Get LLM response
        response = client.generate(prompt)
        if not response:
            logging.error(f"No response generated for {category}")
            continue
            
        logging.info("Raw Response:")
        print(response)
        
        # Clean response
        cleaned_lines = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '##')):
                # Remove numbering and markdown
                line = re.sub(r'^[\d\.\s\*]+', '', line)
                line = re.sub(r'\*\*', '', line)
                cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Validate format
        if not ProductResponseParser.validate_format(cleaned_response):
            logging.error(f"Invalid format for {category}")
            continue
            
        # Parse products
        products = []
        for line in cleaned_lines:
            if product := ProductResponseParser.parse_product_line(line, category):
                products.append(product)
        
        logging.info(f"\nParsed {len(products)} products:")
        for product in products:
            print(f"\n{product}")

if __name__ == "__main__":
    test_product_generation()