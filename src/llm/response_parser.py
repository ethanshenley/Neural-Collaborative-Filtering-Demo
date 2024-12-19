# src/llm/response_parser.py

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging
import re
from datetime import datetime
import uuid

@dataclass
class ParsedProduct:
    """Represents a parsed product with all its attributes"""
    name: str
    size: str
    customizations: List[str]
    dietary_info: List[str]
    attributes: Dict[str, any]

class ProductResponseParser:
    """Parser for LLM-generated product descriptions"""
    
    # Class-level constants
    HOT_KEYWORDS = {
        'hot', 'grilled', 'warm', 'heated', 'fried', 'baked', 'roasted', 
        'toasted', 'steamed', 'pot pie', 'meatloaf', 'melted', 'burger',
        'pizza', 'quesadilla'
    }
    
    COLD_KEYWORDS = {
        'cold', 'chilled', 'fresh', 'iced', 'frozen', 'cool', 'refrigerated',
        'salad', 'sushi', 'parfait', 'smoothie'
    }
    
    DIETARY_PATTERNS = {
        'Vegetarian': [r'vegetarian', r'veggie', r'meatless'],
        'Vegan': [r'vegan', r'plant.?based', r'dairy.?free'],
        'Gluten-Free': [r'gluten.?free', r'\bGF\b'],
        'Low-Carb': [r'low.?carb', r'keto', r'carb.?smart'],
        'Dairy-Free': [r'dairy.?free', r'non.?dairy', r'lactose.?free']
    }

    @classmethod
    def clean_product_name(cls, name: str) -> str:
        """Clean product name removing numbers, bullets, and formatting"""
        # Remove leading numbers and dots
        name = re.sub(r'^[\d\.\*\s]+', '', name)
        # Remove any markdown
        name = re.sub(r'\*+', '', name)
        # Remove section headers
        name = re.sub(r'^#+\s*.*?:\s*', '', name)
        # Clean extra whitespace
        return ' '.join(name.split()).strip()

    @classmethod
    def standardize_size(cls, size_str: str) -> str:
        """Standardize size format"""
        size = size_str.strip()
        
        # Handle standard mappings
        size_mapping = {
            'med': 'Medium',
            'lg': 'Large',
            'sm': 'Small',
            'regular': 'Regular',
            'reg': 'Regular',
            'ind': 'Individual'
        }
        
        size_lower = size.lower()
        if size_lower in size_mapping:
            return size_mapping[size_lower]
            
        # Handle special cases (e.g., "1/3 lb", "12-inch")
        if any(unit in size_lower for unit in ['lb', 'oz', 'inch']):
            return size
            
        return size

    @classmethod
    def extract_customizations(cls, text: str) -> List[str]:
        """Extract and clean customization options"""
        customizations = []
        
        # Extract content between brackets
        bracket_match = re.search(r'\[(.*?)\]', text)
        if bracket_match:
            # Split on commas, handling nested parentheses
            items = []
            current = []
            nesting_level = 0
            
            for char in bracket_match.group(1) + ',':
                if char == '(':
                    nesting_level += 1
                    current.append(char)
                elif char == ')':
                    nesting_level -= 1
                    current.append(char)
                elif char == ',' and nesting_level == 0:
                    items.append(''.join(current))
                    current = []
                else:
                    current.append(char)
                    
            # Clean each item
            for item in items:
                item = item.strip()
                # Skip dietary info and temperature indicators
                if item and not any(diet.lower() in item.lower() for diet in 
                    ['vegetarian', 'vegan', 'gluten-free', 'cold', 'hot']):
                    customizations.append(item)
                    
        return [c.strip() for c in customizations if c.strip()]

    @classmethod
    def extract_dietary_info(cls, text: str) -> List[str]:
        """Extract dietary information"""
        dietary_info = set()
        text_lower = text.lower()
        
        # Check each dietary pattern
        for diet, patterns in cls.DIETARY_PATTERNS.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                dietary_info.add(diet)
                if re.search(rf'{patterns[0]}.*?option', text_lower):
                    dietary_info.add(f"{diet} Option Available")
                    
        return sorted(list(dietary_info))

    @classmethod
    def infer_temperature(cls, text: str, category: str) -> Optional[str]:
        """Infer temperature from description and category"""
        text_lower = text.lower()
        
        # Explicit temperature check
        if any(kw in text_lower for kw in cls.HOT_KEYWORDS):
            return 'Hot'
        if any(kw in text_lower for kw in cls.COLD_KEYWORDS):
            return 'Cold'
            
        # Category-based defaults
        category_temps = {
            'HOT': 'Hot',
            'RTE': 'Cold',
            'BEV': 'Cold'  # Default for beverages unless specified hot
        }
        
        return category_temps.get(category)

    @classmethod
    def parse_product_line(cls, line: str, category: str) -> Optional[Dict]:
        """Parse a single product line into schema-compatible format"""
        try:
            # Extract components with regex
            pattern = r'^([^()]+)\(([^)]+)\)\s*\[Department:(\w+),\s*Category:(\w+),\s*([^\]]+)\]'
            match = re.match(pattern, line.strip())
            
            if not match:
                return None
                
            name, size, dept_id, cat_id, attributes = match.groups()
            
            # Create schema-compatible product record
            return {
                "product_id": f"P{uuid.uuid4().hex[:8].upper()}",
                "category_id": cat_id,
                "department_id": dept_id,
                "product_name": f"{name.strip()} ({size.strip()})",
                "last_modified_date": datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Error parsing line: {line}\nError: {str(e)}")
            return None

    @classmethod
    def validate_format(cls, response: str) -> bool:
        """Validate response format"""
        if not response or not response.strip():
            return False
            
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        pattern = r'^([^()]+)\(([^)]+)\)\s*\[Department:(\w+),\s*Category:(\w+),\s*([^\]]+)\]$'
        
        for line in lines:
            if not re.match(pattern, line):
                return False
        return True

    @classmethod
    def parse_llm_response(cls, response: str, category: Optional[str] = None) -> List[ParsedProduct]:
        """Parse the complete LLM response"""
        products = []
        
        # Split into lines and process each valid line
        lines = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.startswith('#')]
        
        for line in lines:
            if product := cls.parse_product_line(line, category):
                products.append(product)
                
        return products

    @staticmethod
    def to_dict(parsed_product: ParsedProduct) -> Dict:
        """Convert ParsedProduct to dictionary format"""
        return {
            "product_name": parsed_product.name,
            "size": parsed_product.size,
            "customizations": parsed_product.customizations,
            "dietary_info": parsed_product.dietary_info,
            "temperature": parsed_product.attributes.get('temperature'),
            "preparation": parsed_product.attributes.get('preparation'),
            "category": parsed_product.attributes.get('category')
        }