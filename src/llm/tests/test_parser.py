# test_parser_comprehensive.py

import logging
import json
from src.llm.response_parser import ProductResponseParser
from src.llm.vertex_client import VertexLLMClient

def print_parsed_product(product_dict: dict, indent: int = 2):
    """Pretty print a parsed product"""
    print(json.dumps(product_dict, indent=indent))

def test_parser_components():
    """Test individual parser components"""
    parser = ProductResponseParser()
    
    print("\n=== Testing Name Cleaning ===")
    test_names = [
        "1. **Spicy Chicken Sandwich**",
        "## Menu Item: Buffalo Wings",
        "**3. Deluxe Burger (with cheese)**",
        "* Premium Pizza"
    ]
    for name in test_names:
        cleaned = parser.clean_product_name(name)
        print(f"Original: {name}")
        print(f"Cleaned:  {cleaned}\n")

    print("\n=== Testing Size Standardization ===")
    test_sizes = [
        "Regular/Large",
        "12-inch",
        "Med Size",
        "lg",
        "Individual/Sharing"
    ]
    for size in test_sizes:
        standardized = parser.standardize_size(size)
        print(f"Original: {size}")
        print(f"Standardized: {standardized}\n")

    print("\n=== Testing Dietary Info Extraction ===")
    test_descriptions = [
        "Vegetarian option available with tofu",
        "Gluten-free bun available",
        "Vegan and dairy-free",
        "Low-carb friendly, keto option"
    ]
    for desc in test_descriptions:
        dietary = parser.extract_dietary_info(desc)
        print(f"Description: {desc}")
        print(f"Extracted: {dietary}\n")

def test_with_llm():
    """Test parser with actual LLM responses"""
    client = VertexLLMClient(project_id="ce-demo-space")
    parser = ProductResponseParser()
    
    # Test prompts for different categories
    test_prompts = {
        "MTO": """Generate a made-to-order sandwich with multiple options and dietary alternatives.""",
        "BEV": """Generate a specialty beverage with size options and dietary information.""",
        "HOT": """Generate a hot prepared food item with customization options."""
    }
    
    print("\n=== Testing with LLM Generated Content ===")
    for category, prompt in test_prompts.items():
        print(f"\nTesting {category} Category:")
        response = client.generate(prompt)
        if response:
            print("\nLLM Response:")
            print(response)
            
            parsed_products = parser.parse_llm_response(response, category)
            
            print("\nParsed Results:")
            for i, product in enumerate(parsed_products, 1):
                print(f"\nProduct {i}:")
                print_parsed_product(parser.to_dict(product))

def test_complex_cases():
    """Test parser with complex edge cases"""
    parser = ProductResponseParser()
    
    print("\n=== Testing Complex Cases ===")
    
    test_cases = [
        """Mega Burger Deluxe (Large) [Brioche Bun, Double Beef Patty (8oz total), 
        American & Swiss Cheese, Lettuce/Tomato/Onion/Pickles, Special Sauce, 
        Gluten-free bun available] - Prepared hot and fresh""",
        
        """Health Bowl (Regular) [Choose Base: Quinoa or Brown Rice, 
        Protein Options (Grilled Chicken, Tofu, or Falafel), 
        Any 4 Vegetables, Choice of Dressing] Vegetarian/Vegan Options Available""",
        
        """Ultimate Coffee Experience (16oz/24oz) [Premium Roast, 
        Choice of Milk (Whole, 2%, Almond, Oat), 
        Flavoring Options (Vanilla, Caramel, Hazelnut), 
        Temperature: Hot or Iced]"""
    ]
    
    for case in test_cases:
        print("\nTest Case:")
        print(case)
        
        result = parser.parse_product_line(case)
        if result:
            print("\nParsed Result:")
            print_parsed_product(parser.to_dict(result))
        else:
            print("Failed to parse")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=== Starting Comprehensive Parser Tests ===")
    
    # Test individual components
    test_parser_components()
    
    # Test complex cases
    test_complex_cases()
    
    # Test with LLM
    test_with_llm()

if __name__ == "__main__":
    main()