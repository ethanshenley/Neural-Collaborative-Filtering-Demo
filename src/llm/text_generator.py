# src/llm/text_generator.py

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

@dataclass
class CategoryPrompt:
    """Template for category-specific product naming"""
    context: str
    examples: str
    attributes: List[str]

class ProductTextGenerator:
    def __init__(self, llm_client):
        """Initialize the text generator with an LLM client"""
        self.llm_client = llm_client
        self._initialize_prompts()
        
    def _initialize_prompts(self):
        """Initialize category-specific prompting templates"""
        self.category_prompts = {
            "MTO": CategoryPrompt(
                context="Made-to-order food items at Sheetz convenience stores, freshly prepared in the kitchen.",
                examples="1. Spicy Chicken Sandwich with Pepper Jack\n2. Build-Your-Own Breakfast Burrito",
                attributes=["size", "customizations", "dietary_info"]
            ),
            "RTE": CategoryPrompt(
                context="Ready-to-eat prepared foods, available for immediate consumption.",
                examples="1. Classic Turkey & Swiss Wrap\n2. Chef's Salad with Ranch",
                attributes=["preparation", "serving_size", "dietary_info"]
            ),
            "BEV": CategoryPrompt(
                context="Packaged beverages including sodas, energy drinks, and water.",
                examples="1. Mountain Dew (20oz)\n2. Pure Life Spring Water (1L)",
                attributes=["brand", "size", "container_type"]
            ),
            # Add more categories as needed
        }

    def _get_prompt_for_products(self, 
                               products: List[Dict],
                               category_code: str) -> str:
        """Generate prompt for a batch of products in the same category"""
        prompt_template = self.category_prompts.get(
            category_code,
            CategoryPrompt(
                context="Products sold at Sheetz convenience stores.",
                examples="",
                attributes=["brand", "size"]
            )
        )

        return f"""Generate realistic product names for {len(products)} items at Sheetz convenience store.

Category Context:
{prompt_template.context}

Required Attributes:
{', '.join(prompt_template.attributes)}

Example Format:
{prompt_template.examples}

Generate {len(products)} unique product names that would realistically be found at Sheetz.
Each product should be specific and include relevant attributes in parentheses.

Format each product name on a new line."""

    def enrich_products(self, 
                       products: List[Dict], 
                       batch_size: int = 10) -> List[Dict]:
        """
        Enrich products with LLM-generated names and descriptions
        
        Args:
            products: List of product dictionaries with structural information
            batch_size: Number of products to process in each LLM call
        """
        # Group products by category for context consistency
        products_by_category = {}
        for product in products:
            cat_code = product['category_id']  # You might need to adjust this
            if cat_code not in products_by_category:
                products_by_category[cat_code] = []
            products_by_category[cat_code].append(product)

        enriched_products = []
        
        # Process each category
        for category_code, category_products in tqdm(products_by_category.items()):
            # Process in batches
            for i in range(0, len(category_products), batch_size):
                batch = category_products[i:i + batch_size]
                
                # Generate prompt for this batch
                prompt = self._get_prompt_for_products(batch, category_code)
                
                try:
                    # Get product names from LLM
                    response = self.llm_client.generate(prompt)
                    product_names = [name.strip() for name in response.split('\n') if name.strip()]
                    
                    # Match generated names with products
                    for product, name in zip(batch, product_names):
                        product_copy = product.copy()
                        product_copy['product_name'] = name
                        enriched_products.append(product_copy)
                        
                except Exception as e:
                    logging.error(f"Error enriching products for category {category_code}: {str(e)}")
                    # Fall back to placeholder names if LLM fails
                    for product in batch:
                        product_copy = product.copy()
                        product_copy['product_name'] = f"[{category_code}_PRODUCT]"
                        enriched_products.append(product_copy)

        return enriched_products

    def validate_product_names(self, products: List[Dict]) -> List[Dict]:
        """Optional: Validate and fix any problematic product names"""
        validated_products = []
        for product in products:
            # Add any validation logic here
            # For example, check for inappropriate content, verify format, etc.
            validated_products.append(product)
        return validated_products