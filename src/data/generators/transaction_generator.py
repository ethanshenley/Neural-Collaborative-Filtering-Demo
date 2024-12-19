# src/data/generators/transaction_generator.py

import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

class TransactionGenerator:
    def __init__(self, bq_loader):
        self.bq_loader = bq_loader
        self.customers = self._load_customers()
        self.products = self._load_products()
        
        # Product category relationships (items commonly bought together)
        self.PRODUCT_AFFINITIES = {
            'MTO': ['BEV', 'SNK'],      # People buying food often get drinks and snacks
            'HOT': ['BEV', 'SNK'],      # Same for hot food
            'BEV': ['SNK', 'CND'],      # Drinks often bought with snacks
            'SNK': ['BEV', 'CND'],      # Snacks often with drinks
            'TOB': ['BEV', 'SNK'],      # Tobacco often with drinks
            'ALC': ['SNK', 'BEV']       # Alcohol often with snacks
        }
        
        # Time patterns (24-hour format)
        self.HOURLY_WEIGHTS = {
            0: 0.2,  1: 0.1,  2: 0.1,  3: 0.1,  # Late night
            4: 0.3,  5: 0.8,  6: 1.5,  7: 2.0,  # Morning rush
            8: 1.8,  9: 1.2,  10: 1.0, 11: 1.5, # Late morning
            12: 2.0, 13: 1.5, 14: 1.0, 15: 1.2, # Afternoon
            16: 1.8, 17: 2.0, 18: 1.8, 19: 1.5, # Evening rush
            20: 1.2, 21: 0.8, 22: 0.5, 23: 0.3  # Night
        }
        
        # Product pricing ranges by category
        self.PRICE_RANGES = {
            'MTO': (6.99, 15.99),
            'HOT': (4.99, 12.99),
            'BEV': (1.99, 4.99),
            'SNK': (1.49, 5.99),
            'CND': (0.99, 3.99),
            'TOB': (7.99, 12.99),
            'ALC': (8.99, 24.99)
        }
        
    def _load_customers(self) -> List[Dict]:
        """Load active customers from BigQuery"""
        query = """
        SELECT 
            cardnumber,
            state,
            IFNULL(lifetime_points/NULLIF(DATE_DIFF(CURRENT_DATE(), DATE(activation_date), DAY), 0), 0) as avg_points_per_day,
            enrollment_status,
            first_transaction_location_id
        FROM `sheetz-poc.sheetz_data.loyalty_customer_dim`
        WHERE enrollment_status = 1  -- Only active customers
        """
        return list(self.bq_loader.execute_query(query))
        
    def _load_products(self) -> List[Dict]:
        """Load product catalog from BigQuery"""
        query = """
        SELECT 
            product_id,
            category_id,
            department_id,
            product_name
        FROM `sheetz-poc.sheetz_data.product_features`
        """
        return list(self.bq_loader.execute_query(query))

    def _get_random_time(self, date: datetime) -> datetime:
        """Get random time based on hour weights"""
        hour_weights = list(self.HOURLY_WEIGHTS.values())
        hour = random.choices(range(24), weights=hour_weights)[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return date.replace(hour=hour, minute=minute, second=second)

    def _get_price_for_product(self, product: Dict) -> float:
        """Calculate price for a product"""
        category = product['category_id'][:3]
        price_range = self.PRICE_RANGES.get(category, (1.99, 9.99))
        return round(random.uniform(*price_range), 2)

    def _get_store_for_customer(self, customer: Dict) -> int:
        """Get likely store for customer"""
        # 80% chance they visit their usual store
        if random.random() < 0.8:
            return int(customer['first_transaction_location_id'].replace('S', ''))
        # Otherwise random store in their state
        return random.randint(1000, 9999)

    def _select_products(self, num_items: int, customer: Dict) -> List[Dict]:
        """Select products for transaction based on affinities"""
        products = []
        
        # Select primary product
        primary = random.choice(self.products)
        products.append(primary)
        
        # Add related products based on affinities
        primary_cat = primary['category_id'][:3]
        if primary_cat in self.PRODUCT_AFFINITIES:
            related_cats = self.PRODUCT_AFFINITIES[primary_cat]
            for _ in range(num_items - 1):
                if random.random() < 0.7:  # 70% chance of related product
                    related_cat = random.choice(related_cats)
                    related_products = [p for p in self.products if p['category_id'].startswith(related_cat)]
                    if related_products:
                        products.append(random.choice(related_products))
                else:
                    products.append(random.choice(self.products))
        
        # Fill remaining items with random products
        while len(products) < num_items:
            products.append(random.choice(self.products))
            
        return products

    def _generate_header(self, customer: Dict, date: datetime, 
                        store_number: int, transaction_number: int) -> Dict:
        """Generate transaction header"""
        physical_time = self._get_random_time(date)
        
        return {
            "store_number": store_number,
            "close_date": (date + timedelta(days=1)).date(),
            "transaction_number": transaction_number,
            "business_date": date.date(),
            "physical_date": physical_time.date(),
            "physical_date_time": physical_time,
            "physical_time": physical_time.strftime("%H:%M:%S"),
            "printed": "Y",
            "employee_code": str(random.randint(1000, 9999)),
            "drawer_number": random.randint(1, 6),
            "station_number": random.randint(1, 8),
            "shift_number": (physical_time.hour // 8) + 1,
            "bill_status": "C",  # Completed
            "cust_code": customer['cardnumber'],
            "last_modified_date": datetime.now()
        }

    def _generate_items(self, products: List[Dict], transaction_number: int, 
                       store_number: int, date: datetime) -> List[Dict]:
        """Generate transaction body items"""
        items = []
        
        for i, product in enumerate(products, 1):
            quantity = random.randint(1, 3)
            unit_price = self._get_price_for_product(product)
            
            item = {
                "store_number": store_number,
                "close_date": (date + timedelta(days=1)).date(),
                "business_date": date.date(),
                "physical_date": date.strftime("%Y-%m-%d"),
                "transaction_number": transaction_number,
                "line_number": i,
                "inventory_code": product['product_id'],
                "sold_quantity": quantity,
                "unit_cost": round(unit_price * 0.6, 2),  # 40% margin
                "unit_retail": unit_price,
                "extended_cost": round(quantity * unit_price * 0.6, 2),
                "extended_retail": round(quantity * unit_price, 2),
                "inventory_type": "MERCH",
                "department_id": product['department_id'],
                "category_id": product['category_id'],
                "bill_status": "C",
                "scanned": "Y",
                "last_modified_date": datetime.now()
            }
            
            items.append(item)
            
        return items

    def generate_transaction(self, customer: Dict, date: datetime, 
                           transaction_number: int) -> Tuple[Dict, List[Dict]]:
        """Generate a complete transaction"""
        # Determine store and number of items
        store_number = self._get_store_for_customer(customer)
        num_items = np.random.poisson(2.5)  # Base is 2-3 items
        num_items = min(max(1, num_items), 8)  # Between 1 and 8 items
        
        # Select products
        products = self._select_products(num_items, customer)
        
        # Generate header and items
        header = self._generate_header(customer, date, store_number, transaction_number)
        items = self._generate_items(products, transaction_number, store_number, date)
        
        # Calculate totals
        total_retail = sum(item['extended_retail'] for item in items)
        total_cost = sum(item['extended_cost'] for item in items)
        
        # Update header with totals
        header.update({
            "sub_total": round(total_retail, 2),
            "tax_total": round(total_retail * 0.06, 2),  # 6% tax
            "discount": 0.0,  # Could add promotions later
            "change": 0.0,
            "refund": 0.0
        })
        
        return header, items

    def generate_batch(self, start_date: datetime, end_date: datetime, 
                    transactions_per_day: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate transactions for a date range"""
        all_headers = []
        all_items = []
        transaction_number = 1
        
        num_days = (end_date - start_date).days + 1
        with tqdm(total=num_days * transactions_per_day, desc="Generating transactions") as pbar:
            current_date = start_date
            while current_date <= end_date:
                # Generate daily transactions
                for _ in range(transactions_per_day):
                    customer = random.choice(self.customers)
                    header, items = self.generate_transaction(customer, current_date, transaction_number)
                    
                    all_headers.append(header)
                    all_items.extend(items)
                    transaction_number += 1
                    pbar.update(1)
                
                current_date += timedelta(days=1)
                
        return all_headers, all_items