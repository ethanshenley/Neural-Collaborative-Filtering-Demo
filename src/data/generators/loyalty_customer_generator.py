from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, List, Optional
from faker import Faker
import logging

class LoyaltyCustomerGenerator:
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        random.seed(seed)
        Faker.seed(seed)
        
        # Constants for realistic distribution
        self.STATE_DISTRIBUTION = {
            'PA': 0.40,  # Primary market
            'OH': 0.15,
            'WV': 0.15,
            'VA': 0.15,
            'MD': 0.10,
            'NC': 0.05
        }
        
        self.ENROLLMENT_STATUS_DISTRIBUTION = {
            1: 0.80,  # Enrolled
            2: 0.10,  # On Hold
            0: 0.05,  # Unenrolled
            9: 0.05   # Cancelled
        }
        
        self.TRIGGER_GROUPS = {
            1: "High Value Customer",
            2: "Regular Customer",
            3: "Occasional Customer",
            4: "New Customer",
            5: "At-Risk Customer"
        }
        
    def _generate_location_id(self) -> str:
        """Generate a realistic store location ID"""
        return f"S{random.randint(1000, 9999)}"
        
    def _generate_dates(self) -> Dict[str, Optional[datetime]]:
        """Generate coherent dates for customer timeline"""
        now = datetime.now()
        max_history = 1825  # 5 years max history
        
        # Generate activation date
        days_ago = random.randint(1, max_history)
        activation_date = now - timedelta(days=days_ago)
        
        # 5% chance of cancellation
        cancellation_date = None
        if random.random() < 0.05 and days_ago > 30:  # Ensure we have enough days
            cancel_days = random.randint(30, days_ago)
            cancellation_date = activation_date + timedelta(days=cancel_days)
        
        # Generate last purchase within valid range
        last_date = cancellation_date or now
        days_since_activation = max(0, (last_date - activation_date).days)
        last_purchase_date = activation_date + timedelta(
            days=random.randint(0, days_since_activation)
        )
        
        # 80% chance of redemption if active
        last_redemption_date = None
        if not cancellation_date and random.random() < 0.8:
            days_since_purchase = max(0, (now - last_purchase_date).days)
            redemption_days = min(90, days_since_purchase)
            if redemption_days > 0:
                last_redemption_date = last_purchase_date - timedelta(
                    days=random.randint(0, redemption_days)
                )
            else:
                last_redemption_date = last_purchase_date
                
        return {
            "activation_date": activation_date,
            "cancellation_date": cancellation_date,
            "last_purchase_date": last_purchase_date,
            "last_redemption_date": last_redemption_date
        }

    def generate_customer(self) -> Dict:
        """Generate a single customer record"""
        dates = self._generate_dates()
        
        # Calculate lifetime metrics
        days_active = (dates["last_purchase_date"] - dates["activation_date"]).days
        lifetime_points = int(random.gauss(days_active * 10, days_active * 2))
        dollars_redeemed = int(lifetime_points * 0.01 * random.uniform(0.3, 0.8))
        
        # Generate location info
        state = random.choices(
            list(self.STATE_DISTRIBUTION.keys()),
            list(self.STATE_DISTRIBUTION.values())
        )[0]
        
        # Basic customer info
        gender = random.choice([1, 2])  # 1=Female, 2=Male
        age = random.randint(18, 85)
        birth_date = datetime.now() - timedelta(days=age*365)
        
        return {
            "created_at": dates["activation_date"],
            "updated_at": datetime.now(),
            "cardnumber": str(uuid.uuid4().int)[:16],
            "first_name": self.fake.first_name_female() if gender == 1 else self.fake.first_name_male(),
            "middle_name": self.fake.first_name() if random.random() < 0.3 else None,
            "last_name": self.fake.last_name(),
            "address1": self.fake.street_address(),
            "address2": self.fake.secondary_address() if random.random() < 0.2 else None,
            "city": self.fake.city(),
            "state": state,
            "zip": self.fake.zipcode_in_state(state),
            "country": "USA",
            "phone": self.fake.phone_number(),
            "mobile": self.fake.phone_number() if random.random() < 0.8 else None,
            "activation_date": dates["activation_date"].date(),
            "cancellation_date": dates["cancellation_date"].date() if dates["cancellation_date"] else None,
            "last_purchase_date": dates["last_purchase_date"].date(),
            "last_redemption_date": dates["last_redemption_date"].date() if dates["last_redemption_date"] else None,
            "enrollment_status": random.choices(
                list(self.ENROLLMENT_STATUS_DISTRIBUTION.keys()),
                list(self.ENROLLMENT_STATUS_DISTRIBUTION.values())
            )[0],
            "birth_date": birth_date.date(),
            "opt_in": random.choice(['Y', 'N']),
            "gender": gender,
            "drivers_license": f"{state}{random.randint(10000000, 99999999)}",
            "mothers_maiden": self.fake.last_name(),
            "parent_card_number": None,
            "trigger_group": random.randint(1, 5),
            "first_transaction_location_id": self._generate_location_id(),
            "first_transaction_date": dates["activation_date"].strftime("%Y-%m-%d"),
            "dollars_redeemed": dollars_redeemed,
            "lifetime_points": lifetime_points,
            "first_transaction_cashier": random.randint(1000, 9999),
            "location_enrolled": self._generate_location_id(),
            "age": age,
            "business_date": datetime.now().strftime("%Y-%m-%d"),
            "load_date": datetime.now(),
            "file_path": f"gs://sheetz-data/loyalty/{datetime.now().strftime('%Y/%m/%d')}/batch_{uuid.uuid4().hex[:8]}.parquet"
        }

    def generate_batch(self, size: int) -> List[Dict]:
        """Generate a batch of customers"""
        return [self.generate_customer() for _ in range(size)]