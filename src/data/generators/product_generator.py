from datetime import datetime, timedelta
import random
from typing import Dict, List, Set, Tuple
import uuid
from dataclasses import dataclass
from enum import Enum

class DepartmentType(Enum):
    FOOD_SERVICE = "FS"
    BEVERAGES = "BV"
    SNACKS = "SN"
    GROCERY = "GR"
    TOBACCO = "TB"
    ALCOHOL = "AL"
    HEALTH_BEAUTY = "HB"
    AUTO = "AT"
    GENERAL_MERCH = "GM"

@dataclass
class CategoryConfig:
    name: str
    code: str
    avg_products: int  # Average number of products in this category
    product_name_template: str  # Template for product names, will be filled by LLM

class ProductFeatureGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the product feature generator with consistent seeds"""
        self.seed = seed
        random.seed(seed)
        
        # Initialize department configurations
        self._initialize_departments()
        
    def _initialize_departments(self):
        """Initialize department and category configurations"""
        self.DEPARTMENT_CATEGORIES = {
            DepartmentType.FOOD_SERVICE: [
                CategoryConfig("Made To Order", "MTO", 50, "[MTO_PRODUCT]"),
                CategoryConfig("Ready To Eat", "RTE", 40, "[RTE_PRODUCT]"),
                CategoryConfig("Hot Foods", "HOT", 30, "[HOT_FOOD_PRODUCT]"),
                CategoryConfig("Cold Foods", "CLD", 35, "[COLD_FOOD_PRODUCT]")
            ],
            DepartmentType.BEVERAGES: [
                CategoryConfig("Fountain Drinks", "FTN", 20, "[FOUNTAIN_DRINK]"),
                CategoryConfig("Coffee", "COF", 25, "[COFFEE_PRODUCT]"),
                CategoryConfig("Packaged Beverages", "BEV", 150, "[BEVERAGE_PRODUCT]"),
                CategoryConfig("Energy Drinks", "NRG", 75, "[ENERGY_DRINK]"),
                CategoryConfig("Water", "H2O", 40, "[WATER_PRODUCT]")
            ],
            DepartmentType.SNACKS: [
                CategoryConfig("Chips", "CHP", 200, "[CHIPS_PRODUCT]"),
                CategoryConfig("Candy", "CND", 250, "[CANDY_PRODUCT]"),
                CategoryConfig("Nuts & Seeds", "NUT", 100, "[NUTS_PRODUCT]"),
                CategoryConfig("Jerky", "JRK", 50, "[JERKY_PRODUCT]")
            ],
            DepartmentType.GROCERY: [
                CategoryConfig("Packaged Foods", "PKG", 300, "[PACKAGED_FOOD]"),
                CategoryConfig("Dairy", "DRY", 75, "[DAIRY_PRODUCT]"),
                CategoryConfig("Frozen Foods", "FRZ", 100, "[FROZEN_PRODUCT]"),
                CategoryConfig("Basic Grocery", "GRC", 150, "[GROCERY_PRODUCT]")
            ],
            DepartmentType.TOBACCO: [
                CategoryConfig("Cigarettes", "CIG", 400, "[CIGARETTE_PRODUCT]"),
                CategoryConfig("Other Tobacco", "TOB", 150, "[TOBACCO_PRODUCT]"),
                CategoryConfig("E-Cigarettes", "ECG", 100, "[ECIG_PRODUCT]")
            ],
            DepartmentType.ALCOHOL: [
                CategoryConfig("Beer", "BER", 300, "[BEER_PRODUCT]"),
                CategoryConfig("Wine", "WIN", 200, "[WINE_PRODUCT]"),
                CategoryConfig("Malt Beverages", "MLT", 100, "[MALT_PRODUCT]")
            ],
            DepartmentType.HEALTH_BEAUTY: [
                CategoryConfig("OTC Medicine", "OTC", 150, "[OTC_PRODUCT]"),
                CategoryConfig("Personal Care", "PCA", 200, "[PERSONAL_CARE]"),
                CategoryConfig("Health Aids", "HLA", 100, "[HEALTH_AID]")
            ],
            DepartmentType.AUTO: [
                CategoryConfig("Motor Oil", "OIL", 75, "[MOTOR_OIL]"),
                CategoryConfig("Auto Supplies", "AUT", 150, "[AUTO_SUPPLY]"),
                CategoryConfig("Auto Fluids", "FLD", 50, "[AUTO_FLUID]")
            ],
            DepartmentType.GENERAL_MERCH: [
                CategoryConfig("Electronics", "ELC", 100, "[ELECTRONICS]"),
                CategoryConfig("Seasonal", "SSN", 150, "[SEASONAL]"),
                CategoryConfig("General Merchandise", "GEN", 200, "[GENERAL_MERCH]")
            ]
        }
        
        # Generate consistent IDs for departments
        self.department_ids = {
            dept: f"D{i+1:03d}" for i, dept in enumerate(DepartmentType)
        }
        
        # Generate consistent IDs for categories
        self.category_ids = {}
        counter = 1
        for dept in DepartmentType:
            for category in self.DEPARTMENT_CATEGORIES[dept]:
                self.category_ids[category.code] = f"C{counter:04d}"
                counter += 1

    def _generate_product_id(self) -> str:
        """Generate a unique product ID"""
        return f"P{uuid.uuid4().hex[:8].upper()}"

    def _generate_product_batch(self, 
                              department: DepartmentType,
                              category: CategoryConfig,
                              count: int) -> List[Dict]:
        """Generate a batch of products for a specific category"""
        products = []
        
        for _ in range(count):
            product_id = self._generate_product_id()
            
            product = {
                "product_id": product_id,
                "category_id": self.category_ids[category.code],
                "department_id": self.department_ids[department],
                "product_name": category.product_name_template,  # Will be filled by LLM
                "last_modified_date": datetime.now()
            }
            
            products.append(product)
            
        return products

    def generate_products(self) -> List[Dict]:
        """Generate a complete product catalog"""
        all_products = []
        
        # Generate products for each department and category
        for department in DepartmentType:
            for category in self.DEPARTMENT_CATEGORIES[department]:
                # Add some randomness to product count while maintaining approximate ratios
                count = int(category.avg_products * random.uniform(0.9, 1.1))
                products = self._generate_product_batch(department, category, count)
                all_products.extend(products)
                
        return all_products
    
    def get_department_categories(self) -> Dict[str, List[str]]:
        """Get mapping of department IDs to their category IDs"""
        mapping = {}
        for dept in DepartmentType:
            dept_id = self.department_ids[dept]
            mapping[dept_id] = [
                self.category_ids[cat.code]
                for cat in self.DEPARTMENT_CATEGORIES[dept]
            ]
        return mapping
    
    def get_category_info(self) -> Dict[str, Tuple[str, str]]:
        """Get mapping of category IDs to their department ID and code"""
        mapping = {}
        for dept in DepartmentType:
            for category in self.DEPARTMENT_CATEGORIES[dept]:
                cat_id = self.category_ids[category.code]
                mapping[cat_id] = (self.department_ids[dept], category.code)
        return mapping