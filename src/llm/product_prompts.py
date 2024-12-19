from typing import Dict
import re

class ProductPromptGenerator:
    CATEGORY_EXAMPLES = {
        "BEV": {
            "description": "Beverages including hot, cold, and specialty drinks",
            "required_attributes": ["temperature", "sugar content", "caffeine level", "container"],
            "examples": """
Mountain Dew Code Red (20oz) [Cold, Regular Sugar, High Caffeine, Bottle]
Fresh Brewed Coffee (16oz) [Hot, No Sugar, High Caffeine, Cup]
Strawberry Banana Smoothie (24oz) [Cold, Fresh Fruit, Regular Sugar, Made-to-Order]
"""
        },
        "MTO": {
            "description": "Made-to-order fresh prepared foods",
            "required_attributes": ["base", "proteins", "toppings", "sauces", "dietary options"],
            "examples": """
Classic Italian Sub (12-inch) [Italian Bread, Ham, Salami, Provolone, Lettuce, Tomato, Oil & Vinegar]
Buffalo Chicken Wrap (Large) [Tortilla, Grilled Chicken, Blue Cheese, Lettuce, Buffalo Sauce]
Veggie Power Bowl (Regular) [Brown Rice, Mixed Vegetables, Hummus, Feta, Balsamic, Vegetarian]
"""
        },
        "HOT": {
            "description": "Hot prepared foods kept at serving temperature",
            "required_attributes": ["temperature (must say Hot)", "sides", "sauces", "dietary options"],
            "examples": """
Mac & Cheese Bowl (Large) [Creamy Cheese Sauce, Bread Crumbs, Side of Garlic Bread, Hot]
Chicken Tender Basket (5-piece) [Crispy Breaded, Choice of Sauce, Side of Fries, Hot]
Pizza Slice (Regular) [Pepperoni, Mozzarella, Marinara, Hot, Made Fresh]
"""
        },
        "RTE": {
            "description": "Ready-to-eat prepared foods served cold",
            "required_attributes": ["components", "dressings", "temperature (must say Cold)", "dietary options"],
            "examples": """
Caesar Salad (Individual) [Romaine, Parmesan, Croutons, Caesar Dressing, Cold]
Fruit Cup (Regular) [Mixed Fresh Fruit, Greek Yogurt Dip, Cold]
Turkey Club Wrap (Regular) [White Wrap, Turkey, Bacon, Lettuce, Tomato, Mayo, Cold]
"""
        }
    }

    CATEGORY_RULES = {
        "BEV": """BEVERAGE RULES:
1. MUST specify: Hot or Cold
2. MUST include sugar content (Zero, Low, Regular)
3. MUST include caffeine level if applicable
4. MUST specify container type (Cup, Bottle, Can)
5. Size MUST be in oz (12oz, 16oz, 20oz, 24oz)
""",
        "MTO": """MADE-TO-ORDER RULES:
1. MUST include base item (bread, wrap, bowl)
2. MUST list all major components
3. MUST include customization options
4. MUST note dietary alternatives if available
5. Size MUST be descriptive (Regular, Large, 6-inch, 12-inch)
""",
        "HOT": """HOT FOOD RULES:
1. MUST include 'Hot' in customizations
2. MUST list all included sides
3. MUST specify sauce options
4. MUST include any dietary alternatives
5. Size MUST be clear (Regular, Large, Family)
""",
        "RTE": """READY-TO-EAT RULES:
1. MUST include 'Cold' in customizations
2. MUST list all components
3. NO additional descriptions or notes outside brackets
4. MUST use comma separation in brackets
5. Size MUST be (Individual, Regular, Large)
"""
    }

    DEPARTMENT_MAPPING = {
        "MTO": "FS", "RTE": "FS", "HOT": "FS", "BAK": "FS",
        "BEV": "BV", "COF": "BV", "FTN": "BV",
        "SNK": "SC", "CND": "SC",
        "GRO": "GR", "DRY": "GR",
        "TOB": "TA", "ALC": "TA"
    }

    CATEGORY_CODES = {
        "MTO": ["MTO1", "MTO2", "MTO3"],
        "RTE": ["RTE1", "RTE2"],
        "HOT": ["HOT1", "HOT2"],
        "BAK": ["BAK1", "BAK2"],
        "BEV": ["BEV1", "BEV2", "BEV3"],
        "COF": ["COF1", "COF2"],
        "FTN": ["FTN1", "FTN2"],
        "SNK": ["SNK1", "SNK2", "SNK3"],
        "CND": ["CND1", "CND2"],
        "GRO": ["GRO1", "GRO2"],
        "DRY": ["DRY1"],
        "TOB": ["TOB1", "TOB2"],
        "ALC": ["ALC1", "ALC2"]
    }

    @classmethod
    def generate_prompt(cls, department_id: str, category_id: str, count: int) -> str:
        category_type = category_id[:3]

        CATEGORY_INSTRUCTIONS = {
            "MTO": {
                "description": "Made-to-order fresh prepared foods",
                "size_format": "Regular, Large, 6-inch, 12-inch",
                "example": f"Classic Italian Sub(12-inch) [Department:{department_id}, Category:{category_id}, Italian Bread, Ham, Salami, Provolone, Lettuce, Tomato, Oil & Vinegar]"
            },
            "RTE": {
                "description": "Ready-to-eat prepared foods served cold",
                "size_format": "Individual, Regular, Large",
                "example": f"Chicken Caesar Salad(Regular) [Department:{department_id}, Category:{category_id}, Romaine, Parmesan, Croutons, Caesar Dressing, Cold, Gluten-Free]"
            },
            "HOT": {
                "description": "Hot prepared foods",
                "size_format": "Regular, Large, Family",
                "example": f"Mac & Cheese Bowl(Large) [Department:{department_id}, Category:{category_id}, Creamy Cheese Sauce, Bread Crumbs, Side of Garlic Bread, Hot]"
            },
            "BAK": {
                "description": "Fresh baked goods",
                "size_format": "Single, Regular, Half-Dozen, Dozen",
                "example": f"Glazed Donuts(Half-Dozen) [Department:{department_id}, Category:{category_id}, Fresh, Sweet]"
            },
            "BEV": {
                "description": "Cold beverages",
                "size_format": "12oz, 16oz, 20oz, 24oz",
                "example": f"Mountain Berry Blast(20oz) [Department:{department_id}, Category:{category_id}, Cold, Sugar-Free, Low Caffeine, Bottle]"
            },
            "COF": {
                "description": "Hot coffee and specialty drinks",
                "size_format": "12oz, 16oz, 20oz",
                "example": f"Vanilla Latte(16oz) [Department:{department_id}, Category:{category_id}, Hot, No Sugar, High Caffeine, Cup]"
            },
            "FTN": {
                "description": "Fountain drinks and slushies",
                "size_format": "Small(16oz), Medium(24oz), Large(32oz)",
                "example": f"Blue Raspberry Slushie(24oz) [Department:{department_id}, Category:{category_id}, Cold, Sweet, Cup]"
            },
            "SNK": {
                "description": "Packaged snacks",
                "size_format": "Regular, Family, Party",
                "example": f"Kettle Cooked Chips(Regular) [Department:{department_id}, Category:{category_id}, Crunchy, Salty]"
            },
            "CND": {
                "description": "Candy and sweets",
                "size_format": "Single, Regular, Share Size, Family",
                "example": f"Chocolate Bar(Regular) [Department:{department_id}, Category:{category_id}, Sweet]"
            },
            "GRO": {
                "description": "Grocery items",
                "size_format": "Regular, Family",
                "example": f"White Bread(Regular) [Department:{department_id}, Category:{category_id}, Fresh]"
            },
            "DRY": {
                "description": "Dairy products",
                "size_format": "Small, Regular, Large, Family",
                "example": f"Whole Milk(Regular) [Department:{department_id}, Category:{category_id}, Cold, Fresh]"
            },
            "TOB": {
                "description": "Tobacco products (legal, standard retail items)",
                "size_format": "Single Pack, Carton",
                "example": f"Cigarettes(Single Pack) [Department:{department_id}, Category:{category_id}, Standard]"
            },
            "ALC": {
                "description": "Alcoholic beverages (legally sold retail items)",
                "size_format": "Single, 6-Pack, 12-Pack, Case",
                "example": f"Beer(6-Pack) [Department:{department_id}, Category:{category_id}, Cold]"
            }
        }

        category_info = CATEGORY_INSTRUCTIONS.get(category_type, {
            "description": "Store products",
            "size_format": "Regular, Large",
            "example": f"Generic Product(Regular) [Department:{department_id}, Category:{category_id}, Standard]"
        })

        allowed_cats = ", ".join(cls.CATEGORY_CODES[category_type]) if category_type in cls.CATEGORY_CODES else category_id

        prompt = f"""You are to generate exactly {count} products for Sheetz stores.
        
STRICT REQUIREMENTS:
1. Allowed category: {category_id} (This category must be EXACT. No variations.)
2. Department: {department_id} (No changes allowed.)
3. NO numbering, NO bullet points, NO extra text or commentary.
4. Do not produce any invalid categories or departments. ONLY use Department:{department_id}, Category:{category_id}.
5. Format each product on its own line, with EXACT format:
   ProductName(Size) [Department:{department_id}, Category:{category_id}, Attribute1, Attribute2, ...]
6. Follow the sizing and attribute rules as outlined below.
7. If the product type (based on category_id) requires "Cold" or "Hot", ensure it is included.
8. If you cannot adhere to ALL these rules, produce no output.

Product Type: {category_info['description']}
Size Format: {category_info['size_format']}
Valid Category Code: {category_id} must appear exactly as shown.
DO NOT INVENT NEW CATEGORIES.

Example (follow this EXACT style):
{category_info['example']}

Generate {count} products now, following ALL the rules exactly."""

        return prompt

    @staticmethod
    def validate_format(response: str, category: str) -> bool:
        """Validate that response follows required format"""
        if not response.strip():
            return False

        lines = [l.strip() for l in response.split('\n') if l.strip()]
        pattern = r'^[^()\[\]]+\([^()]+\)\s*\[[^\[\]]+\]$'

        for line in lines:
            if not re.match(pattern, line):
                return False
                
            # Category-specific checks
            lower_line = line.lower()
            if category == 'BEV' and not any(x in lower_line for x in ['oz', 'cup', 'bottle', 'can']):
                return False
            if category == 'HOT' and 'hot' not in lower_line:
                return False
            if category == 'RTE' and 'cold' not in lower_line:
                return False

        return True
