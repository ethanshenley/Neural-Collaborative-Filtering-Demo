# src/llm/category_validator.py

from typing import List, Dict, Optional, Tuple

class CategoryValidator:
    """Validates products against category rules"""
    
    CATEGORY_KEYWORDS = {
        "BEV": {
            'required': ['oz', 'cup', 'bottle', 'can'],
            'typical': ['drink', 'coffee', 'tea', 'soda', 'smoothie', 'juice', 'water'],
            'forbidden': ['sandwich', 'burger', 'salad', 'wrap']
        },
        "HOT": {
            'required': ['hot'],
            'typical': ['burger', 'pizza', 'fries', 'wings', 'mac & cheese'],
            'forbidden': ['cold', 'iced']
        },
        "RTE": {
            'required': ['cold'],
            'typical': ['salad', 'wrap', 'fruit', 'yogurt'],
            'forbidden': ['hot', 'grilled', 'fried']
        },
        "MTO": {
            'typical': ['sandwich', 'burger', 'wrap', 'sub', 'quesadilla'],
            'forbidden': []
        }
    }
    
    @classmethod
    def validate_product(cls, 
                        product: Dict, 
                        category: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a product against category rules
        Returns: (is_valid, error_message)
        """
        if category not in cls.CATEGORY_KEYWORDS:
            return True, None
            
        text = (
            f"{product['product_name']} {' '.join(product['customizations'])}"
        ).lower()
        
        # Check required keywords
        required = cls.CATEGORY_KEYWORDS[category].get('required', [])
        if required and not any(kw in text for kw in required):
            return False, f"Missing required keywords for {category}"
            
        # Check forbidden keywords
        forbidden = cls.CATEGORY_KEYWORDS[category].get('forbidden', [])
        if any(kw in text for kw in forbidden):
            return False, f"Contains forbidden keywords for {category}"
            
        return True, None
        
    @classmethod
    def suggest_category(cls, product: Dict) -> str:
        """Suggest the most appropriate category for a product"""
        text = (
            f"{product['product_name']} {' '.join(product['customizations'])}"
        ).lower()
        
        # Score each category
        scores = {}
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0
            # Required keywords are worth 3 points
            for kw in keywords.get('required', []):
                if kw in text:
                    score += 3
            # Typical keywords are worth 1 point
            for kw in keywords.get('typical', []):
                if kw in text:
                    score += 1
            # Forbidden keywords are worth -2 points
            for kw in keywords.get('forbidden', []):
                if kw in text:
                    score -= 2
            scores[category] = score
            
        # Return category with highest score
        return max(scores.items(), key=lambda x: x[1])[0]