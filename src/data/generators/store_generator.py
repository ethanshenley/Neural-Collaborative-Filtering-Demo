from datetime import datetime, timedelta, date
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class StoreType(Enum):
    STANDARD = "STANDARD"
    TRAVEL_CENTER = "TRAVEL_CENTER"
    EXPRESS = "EXPRESS"
    URBAN = "URBAN"

class StoreStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    CONSTRUCTION = "CONSTRUCTION"
    REMODEL = "REMODEL"
    TEMPORARY_CLOSED = "TEMPORARY_CLOSED"

@dataclass
class StoreLocationProfile:
    """Represents the location characteristics of a store"""
    urban_density: float  # 0-1, higher means more urban
    highway_proximity: float  # 0-1, higher means closer to highway
    income_level: float  # 0-1, higher means higher income area
    competition_density: float  # 0-1, higher means more competition
    residential_density: float  # 0-1, higher means more residential
    
class ComprehensiveStoreGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the store generator with consistent seeds"""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Core business rules
        self.MIN_STORE_NUMBER = 1
        self.MAX_STORE_NUMBER = 9999
        self.STORE_TYPE_WEIGHTS = {
            StoreType.STANDARD: 0.60,
            StoreType.TRAVEL_CENTER: 0.20,
            StoreType.EXPRESS: 0.15,
            StoreType.URBAN: 0.05
        }
        
        # State distribution (based on Sheetz actual market presence)
        self.STATE_WEIGHTS = {
            'PA': 0.40,  # Pennsylvania - primary market
            'OH': 0.15,  # Ohio
            'WV': 0.15,  # West Virginia
            'VA': 0.15,  # Virginia
            'MD': 0.10,  # Maryland
            'NC': 0.05   # North Carolina
        }
        
        # Initialize core lookup tables
        self._initialize_lookup_tables()
        
    def _initialize_lookup_tables(self):
        """Initialize all lookup tables and reference data"""
        # State-specific location bounds
        self.STATE_BOUNDS = {
            'PA': {'lat': (39.7, 42.0), 'lon': (-80.5, -75.0)},
            'OH': {'lat': (38.4, 41.9), 'lon': (-84.8, -80.5)},
            'WV': {'lat': (37.2, 40.6), 'lon': (-82.6, -77.7)},
            'VA': {'lat': (36.5, 39.5), 'lon': (-83.7, -75.2)},
            'MD': {'lat': (37.9, 39.7), 'lon': (-79.5, -75.0)},
            'NC': {'lat': (35.0, 36.5), 'lon': (-84.3, -75.5)}
        }
        
        # Store configuration types by store type
        self.STORE_CONFIG_TYPES = {
            StoreType.STANDARD: [1, 2, 3],
            StoreType.TRAVEL_CENTER: [4, 5],
            StoreType.EXPRESS: [6],
            StoreType.URBAN: [7, 8]
        }
        
    def _generate_store_profile(self, store_type: StoreType) -> StoreLocationProfile:
        """Generate a consistent store profile based on store type"""
        if store_type == StoreType.URBAN:
            return StoreLocationProfile(
                urban_density=random.uniform(0.7, 1.0),
                highway_proximity=random.uniform(0, 0.4),
                income_level=random.uniform(0.3, 1.0),
                competition_density=random.uniform(0.6, 1.0),
                residential_density=random.uniform(0.7, 1.0)
            )
        elif store_type == StoreType.TRAVEL_CENTER:
            return StoreLocationProfile(
                urban_density=random.uniform(0, 0.3),
                highway_proximity=random.uniform(0.8, 1.0),
                income_level=random.uniform(0.2, 0.8),
                competition_density=random.uniform(0.2, 0.6),
                residential_density=random.uniform(0, 0.3)
            )
        elif store_type == StoreType.EXPRESS:
            return StoreLocationProfile(
                urban_density=random.uniform(0.3, 0.7),
                highway_proximity=random.uniform(0.2, 0.6),
                income_level=random.uniform(0.3, 0.8),
                competition_density=random.uniform(0.4, 0.8),
                residential_density=random.uniform(0.4, 0.8)
            )
        else:  # STANDARD
            return StoreLocationProfile(
                urban_density=random.uniform(0.2, 0.8),
                highway_proximity=random.uniform(0.3, 0.7),
                income_level=random.uniform(0.2, 0.9),
                competition_density=random.uniform(0.3, 0.7),
                residential_density=random.uniform(0.3, 0.8)
            )
            
    def _generate_core_identifiers(self) -> Tuple[int, StoreType, StoreLocationProfile]:
        """Generate the core identifying information for a store"""
        store_number = random.randint(self.MIN_STORE_NUMBER, self.MAX_STORE_NUMBER)
        store_type = np.random.choice(
            list(self.STORE_TYPE_WEIGHTS.keys()),
            p=list(self.STORE_TYPE_WEIGHTS.values())
        )
        store_profile = self._generate_store_profile(store_type)
        
        return store_number, store_type, store_profile
        
    def _generate_location_fields(self, 
                                store_type: StoreType, 
                                profile: StoreLocationProfile) -> Dict:
        """Generate all location-related fields"""
        # Select state based on weights
        state = np.random.choice(
            list(self.STATE_WEIGHTS.keys()),
            p=list(self.STATE_WEIGHTS.values())
        )
        
        # Generate lat/long within state bounds
        bounds = self.STATE_BOUNDS[state]
        latitude = round(random.uniform(bounds['lat'][0], bounds['lat'][1]), 6)
        longitude = round(random.uniform(bounds['lon'][0], bounds['lon'][1]), 6)
        
        # Interstate proximity based on store type and profile
        is_interstate = (
            store_type == StoreType.TRAVEL_CENTER or 
            (profile.highway_proximity > 0.7 and random.random() < 0.8)
        )
        
        return {
            "City": "[CITY_PLACEHOLDER]",
            "County": "[COUNTY_PLACEHOLDER]",
            "State": state,
            "Street": "[STREET_ADDRESS_PLACEHOLDER]",
            "Postal_Code": f"{random.randint(10000, 99999)}",
            "Latitude": latitude,
            "Longitude": longitude,
            "Interstate": is_interstate,
            "At_Divided_Highway": is_interstate or random.random() < 0.4,
            "Divided_Side_Road": random.random() < 0.3,
            "Corner_Location": random.random() < (0.7 if profile.urban_density > 0.7 else 0.3),
            "Side_Road_Access": random.choice(["MAIN", "SIDE", "BOTH"]),
            "Main_Road_Access_Type": random.choice(["PRIMARY", "SECONDARY", "TERTIARY"]),
            "Signalized_Intersection": random.random() < 0.6,
            "Municipality": "[MUNICIPALITY_PLACEHOLDER]",
            "Locality": "[LOCALITY_PLACEHOLDER]",
            "Location_Type": store_type.value
        }
# Continuing from previous class...

    def _generate_physical_specs(self, 
                               store_type: StoreType, 
                               profile: StoreLocationProfile) -> Dict:
        """Generate physical specifications of the store"""
        # Base square footage by store type
        base_footage = {
            StoreType.STANDARD: random.randint(3500, 4500),
            StoreType.TRAVEL_CENTER: random.randint(5000, 7000),
            StoreType.EXPRESS: random.randint(2000, 3000),
            StoreType.URBAN: random.randint(2500, 3500)
        }[store_type]
        
        # Adjust for location profile
        footage_multiplier = 1.0
        if profile.urban_density > 0.8:
            footage_multiplier *= random.uniform(0.8, 0.9)  # Urban stores tend to be smaller
        if profile.income_level > 0.8:
            footage_multiplier *= random.uniform(1.1, 1.2)  # Higher income areas tend to have larger stores
            
        store_square_footage = int(base_footage * footage_multiplier)
        
        # Calculate lot square footage (typically 2.5-4x building size)
        lot_multiplier = {
            StoreType.STANDARD: random.uniform(3.0, 4.0),
            StoreType.TRAVEL_CENTER: random.uniform(4.0, 5.0),
            StoreType.EXPRESS: random.uniform(2.5, 3.0),
            StoreType.URBAN: random.uniform(1.5, 2.5)  # Urban locations have tighter lots
        }[store_type]
        
        lot_square_footage = int(store_square_footage * lot_multiplier)
        
        # Calculate parking stalls based on square footage and type
        base_parking = store_square_footage / 250  # Basic retail ratio
        parking_multiplier = {
            StoreType.STANDARD: random.uniform(1.0, 1.2),
            StoreType.TRAVEL_CENTER: random.uniform(1.5, 2.0),
            StoreType.EXPRESS: random.uniform(0.8, 1.0),
            StoreType.URBAN: random.uniform(0.6, 0.8)
        }[store_type]
        
        parking_stalls = int(base_parking * parking_multiplier)
        
        # Generate seating capacity
        if store_type == StoreType.EXPRESS:
            inside_seating = random.randint(0, 8)
            outside_seating = 0
        elif store_type == StoreType.URBAN:
            inside_seating = random.randint(15, 30)
            outside_seating = random.randint(4, 12)
        elif store_type == StoreType.TRAVEL_CENTER:
            inside_seating = random.randint(30, 50)
            outside_seating = random.randint(8, 16)
        else:  # STANDARD
            inside_seating = random.randint(20, 35)
            outside_seating = random.randint(6, 14)
            
        # Occupancy (typically 1 person per 30 sq ft for retail)
        occupancy = int(store_square_footage / 30)
        
        return {
            "Store_Square_Footage": store_square_footage,
            "Lot_Square_Footage": lot_square_footage,
            "Parking_Stalls": parking_stalls,
            "Inside_Seating": inside_seating,
            "OutSide_Seating": outside_seating,
            "Occupancy": occupancy,
            "Building_Type": store_type.value,
            "RTU_Tonnage": round(store_square_footage * 0.00275, 1),  # Typical HVAC sizing
            "Heating_Source": random.choice(["ELECTRIC", "GAS", "HYBRID"])
        }
        
    def _generate_dates(self) -> Dict:
        """Generate all date-related fields with proper temporal logic"""
        # Generate base dates
        current_date = date.today()
        max_age_years = 25  # Maximum age of oldest stores
        
        # Generate open date
        open_date = current_date - timedelta(days=random.randint(365, 365 * max_age_years))
        
        # Pre-live date is typically 2-4 weeks before open date
        pre_live_date = open_date - timedelta(days=random.randint(14, 30))
        
        # Brand refresh happens every 7-10 years
        years_since_open = (current_date - open_date).days / 365
        if years_since_open > 7:
            brand_refresh_date = open_date + timedelta(days=random.randint(7*365, min(years_since_open, 10)*365))
        else:
            brand_refresh_date = None
            
        # Last remodel occurs every 5-7 years
        if years_since_open > 5:
            last_remodel_date = open_date + timedelta(days=random.randint(5*365, min(years_since_open, 7)*365))
        else:
            last_remodel_date = None
            
        # Rebuild typically happens at 15-20 years
        if years_since_open > 15:
            last_rebuild_date = open_date + timedelta(days=random.randint(15*365, min(years_since_open, 20)*365))
        else:
            last_rebuild_date = None
            
        # White block remodel (cosmetic updates) every 3-4 years
        if years_since_open > 3:
            white_block_date = open_date + timedelta(days=random.randint(3*365, min(years_since_open, 4)*365))
        else:
            white_block_date = None
            
        return {
            "Open_Date": open_date,
            "Pre_Live_Date": pre_live_date,
            "Brand_Refresh": brand_refresh_date,
            "Last_Remodel_Date": last_remodel_date,
            "Last_Rebuild_Date": last_rebuild_date,
            "White_Block_Date" : white_block_date
        }
    def _generate_fuel_services(self, 
                                    store_type: StoreType, 
                                    profile: StoreLocationProfile) -> Dict:
            """Generate all fuel service related fields"""
            # Basic fuel service probabilities based on store type
            has_diesel = {
                StoreType.TRAVEL_CENTER: 1.0,    # Always
                StoreType.STANDARD: 0.8,         # Very common
                StoreType.EXPRESS: 0.6,          # Common
                StoreType.URBAN: 0.3             # Less common
            }[store_type]
            
            is_diesel_store = random.random() < has_diesel
            
            # Calculate diesel specifics if applicable
            if is_diesel_store:
                if store_type == StoreType.TRAVEL_CENTER:
                    auto_diesel_count = random.randint(4, 8)
                    truck_diesel_count = random.randint(6, 12)
                    high_flow_count = random.randint(4, 8)
                else:
                    auto_diesel_count = random.randint(2, 4)
                    truck_diesel_count = 0
                    high_flow_count = random.randint(0, 2)
            else:
                auto_diesel_count = 0
                truck_diesel_count = 0
                high_flow_count = 0
                
            # MPD (Multi Product Dispenser) count based on store type
            mpd_count = {
                StoreType.TRAVEL_CENTER: random.randint(12, 24),
                StoreType.STANDARD: random.randint(8, 16),
                StoreType.EXPRESS: random.randint(6, 10),
                StoreType.URBAN: random.randint(4, 8)
            }[store_type]
            
            # Fuel types availability
            return {
                "Diesel": is_diesel_store,
                "Auto_Diesel_Dispenser_Cnt": auto_diesel_count,
                "Truck_Diesel_Lane_Cnt": truck_diesel_count,
                "High_Flow_Auto_Disp_Cnt": high_flow_count,
                "MPD": mpd_count,
                "E0_Gas": random.random() < 0.9,  # Very common
                "E15_Gas": random.random() < 0.7,  # Common
                "E85_Gas": random.random() < 0.3,  # Less common
                "Bulk_DEF": is_diesel_store and random.random() < 0.8,
                "Kerosene": random.random() < 0.4,
                "Propane": random.random() < 0.7,
                "Pump_Activation": True,  # Modern standard
                "Pumps_Closed": False
            }
        
    def _generate_ev_charging(self, 
                            store_type: StoreType, 
                            profile: StoreLocationProfile) -> Dict:
        """Generate EV charging related specifications"""
        # Base probability of having EV charging
        ev_prob = {
            StoreType.TRAVEL_CENTER: 0.8,
            StoreType.STANDARD: 0.5,
            StoreType.EXPRESS: 0.3,
            StoreType.URBAN: 0.6
        }[store_type]
        
        # Adjust for location profile
        ev_prob *= (1 + profile.income_level * 0.5)  # Higher income areas more likely
        ev_prob = min(ev_prob, 1.0)
        
        has_ev = random.random() < ev_prob
        
        if has_ev:
            # Determine charging level and stall count
            if store_type == StoreType.TRAVEL_CENTER:
                stall_count = random.randint(6, 12)
                power_max = 350
            elif profile.income_level > 0.7:
                stall_count = random.randint(4, 8)
                power_max = random.choice([150, 350])
            else:
                stall_count = random.randint(2, 4)
                power_max = random.choice([50, 150])
                
            return {
                "Electric_Vehicle_Charger": True,
                "EV_PARKING_TOTAL_STALLS": stall_count,
                "EV_Charger_Plug_Type": "MULTIPLE",
                "EV_PLUGTYPE_CCS": True,
                "EV_PLUGTYPE_CHADEMO": random.random() < 0.6,
                "EV_PLUGTYPE_J1772": True,
                "EV_PLUGTYPE_NACS": random.random() < 0.3,  # Newer standard
                "EV_POWER_MAX": power_max
            }
        else:
            return {
                "Electric_Vehicle_Charger": False,
                "EV_PARKING_TOTAL_STALLS": 0,
                "EV_Charger_Plug_Type": None,
                "EV_PLUGTYPE_CCS": False,
                "EV_PLUGTYPE_CHADEMO": False,
                "EV_PLUGTYPE_J1772": False,
                "EV_PLUGTYPE_NACS": False,
                "EV_POWER_MAX": 0
            }
            
    def _generate_employee_data(self,
                              store_type: StoreType,
                              square_footage: int,
                              profile: StoreLocationProfile) -> Dict:
        """Generate all employee and staffing related fields"""
        # Base employee count based on store size and type
        base_employees_per_sqft = {
            StoreType.TRAVEL_CENTER: 0.020,
            StoreType.STANDARD: 0.015,
            StoreType.EXPRESS: 0.012,
            StoreType.URBAN: 0.018
        }[store_type]
        
        # Adjust for location factors
        employee_multiplier = 1.0
        if profile.urban_density > 0.8:
            employee_multiplier *= 1.2  # Urban areas need more staff
        if profile.competition_density > 0.8:
            employee_multiplier *= 0.9  # High competition areas run leaner
            
        total_employees = int(square_footage * base_employees_per_sqft * employee_multiplier)
        
        # Calculate full/part time split (typically 30/70)
        full_time = int(total_employees * random.uniform(0.25, 0.35))
        part_time = total_employees - full_time
        
        # Kronos (workforce management) data
        kronos_min = int(total_employees * 0.8)
        kronos_max = int(total_employees * 1.2)
        
        return {
            "Total_Employees": total_employees,
            "Full_Time_Employees": full_time,
            "Part_Time_Employees": part_time,
            "Kronos_Maximum_Headcount": kronos_max,
            "Kronos_Minimum_Headcount": kronos_min,
            "Kronos_Store": "Y",
            "Kronos_Store_ID": random.randint(10000, 99999),
            "Kronos_WTK_Rollout_Date": self.fake.date_between(start_date='-5y', end_date='now'),
            "Manager_Employee_ID": random.randint(10000, 99999),
            "Manager_Name": "[MANAGER_NAME_PLACEHOLDER]",
            "Manager_Sheetz_Experience_Years": random.randint(1, 25),
            "District_Manager": "[DISTRICT_MANAGER_PLACEHOLDER]",
            "District_Manager_Employee_ID": random.randint(10000, 99999),
            "Regional_Manager": "[REGIONAL_MANAGER_PLACEHOLDER]",
            "District_Number": f"D{random.randint(1,99):02d}",
            "Region_Number": f"R{random.randint(1,9)}",
            "Wage_Area": f"WA{random.randint(1,9)}"
        }
    
    def _generate_food_service(self,
                                    store_type: StoreType,
                                    profile: StoreLocationProfile) -> Dict:
            """Generate all food service related fields"""
            # Base tier assignments based on store type and profile
            if store_type == StoreType.EXPRESS:
                mto_tier = "EXPRESS"
                rte_tier = "C"
                coffee_tier = "BASIC"
            elif store_type == StoreType.TRAVEL_CENTER:
                mto_tier = "PREMIUM"
                rte_tier = "A"
                coffee_tier = "PREMIUM"
            else:
                # For STANDARD and URBAN, consider location profile
                if profile.income_level > 0.7:
                    mto_tier = "PREMIUM"
                    rte_tier = "A"
                    coffee_tier = "PREMIUM"
                elif profile.income_level > 0.4:
                    mto_tier = "STANDARD"
                    rte_tier = "B"
                    coffee_tier = "STANDARD"
                else:
                    mto_tier = "BASIC"
                    rte_tier = "C"
                    coffee_tier = "BASIC"

            # Equipment presence based on tiers
            has_premium_equipment = mto_tier == "PREMIUM"
            has_standard_equipment = mto_tier in ["PREMIUM", "STANDARD"]
            
            return {
                "MTO_Tier": mto_tier,
                "RTE_Tier": rte_tier,
                "RTE_SS_Coffee_Tier": coffee_tier,
                "SS_Coffee_Tier": coffee_tier,
                "Coffee_Brewers": coffee_tier,
                "Espresso_Machine": has_premium_equipment,
                "Fryers": "PREMIUM" if has_premium_equipment else "STANDARD" if has_standard_equipment else "BASIC",
                "Ovens": "PREMIUM" if has_premium_equipment else "STANDARD" if has_standard_equipment else "BASIC",
                "Pizza_Oven": 2 if has_premium_equipment else 1 if has_standard_equipment else 0,
                "Big_6_Ice_Cream": "Y" if has_standard_equipment else "N",
                "Coke_Freestyle": has_premium_equipment,
                "SBC_Tier": coffee_tier,  # Specialty Beverage Center
                "Other_Tier": rte_tier,
                "Allow_Pct_Shrink_Grocery": round(random.uniform(0.01, 0.05), 4)
            }

    def _generate_digital_services(self,
                                 store_type: StoreType,
                                 profile: StoreLocationProfile) -> Dict:
        """Generate all digital and technology service fields"""
        # Base probabilities adjusted by store type and location
        has_delivery = (
            store_type != StoreType.EXPRESS and
            profile.residential_density > 0.4 and
            random.random() < 0.8
        )
        
        return {
            "Online_Ordering": True,  # Modern standard
            "Delivery": has_delivery,
            "Door_Dash": has_delivery and random.random() < 0.9,
            "Grub_Hub": has_delivery and random.random() < 0.8,
            "Uber_Eats": has_delivery and random.random() < 0.8,
            "Curbside": random.random() < 0.9,  # Very common now
            "SHCAN_Go": True,  # Modern standard
            "In_Store_Available": True,
            "WiFi": True,  # Standard in all stores
            "SCO_Lanes": self._calculate_sco_lanes(store_type, profile),
            "Access_Points": self._calculate_access_points(store_type),
            "Bitcoin_Kiosk": random.random() < 0.3,
            "Crypto_Currency_Acceptance": "Y" if random.random() < 0.4 else "N",
            "WV_DMV_Kiosk": profile.state == "WV" and random.random() < 0.5
        }

    def _generate_car_services(self,
                             store_type: StoreType,
                             profile: StoreLocationProfile) -> Dict:
        """Generate car wash and automotive service related fields"""
        # Car wash more likely in higher income areas and non-urban locations
        has_carwash = (
            profile.urban_density < 0.7 and
            profile.income_level > 0.4 and
            random.random() < 0.6
        )
        
        if has_carwash:
            carwash_id = random.randint(10000, 99999)
            car_wash_region = random.choice(["NORTH", "SOUTH", "EAST", "WEST"])
        else:
            carwash_id = None
            car_wash_region = None
            
        return {
            "Car_Wash_Brand": has_carwash,
            "Car_Wash_Type": has_carwash,
            "Car_Wash_Region": car_wash_region,
            "Carwash_ICS_Site_ID": carwash_id,
            "Truck_Scale": store_type == StoreType.TRAVEL_CENTER and random.random() < 0.8,
            "Truck_Parking_Spaces": (
                random.randint(20, 50) if store_type == StoreType.TRAVEL_CENTER
                else 0
            ),
            "Showers": store_type == StoreType.TRAVEL_CENTER and random.random() < 0.9
        }

    def _calculate_sco_lanes(self, store_type: StoreType, profile: StoreLocationProfile) -> int:
        """Calculate number of self-checkout lanes based on store characteristics"""
        if store_type == StoreType.EXPRESS:
            return random.randint(0, 2)
        elif store_type == StoreType.TRAVEL_CENTER:
            return random.randint(4, 6)
        elif profile.urban_density > 0.7:
            return random.randint(3, 5)
        else:
            return random.randint(2, 4)

    def _calculate_access_points(self, store_type: StoreType) -> int:
        """Calculate number of network access points needed"""
        base_points = {
            StoreType.TRAVEL_CENTER: random.randint(8, 12),
            StoreType.STANDARD: random.randint(6, 8),
            StoreType.URBAN: random.randint(4, 6),
            StoreType.EXPRESS: random.randint(3, 5)
        }[store_type]
        
        return base_points

    def _generate_maintenance_info(self, store_type: StoreType) -> Dict:
        """Generate maintenance and facilities related information"""
        # Select maintenance area and region
        maint_area = random.randint(1, 9)
        maint_regions = ["EAST", "WEST", "CENTRAL", "NORTH", "SOUTH"]
        maint_geographical_area = random.choice(maint_regions)
        
        return {
            "Maint_Area": maint_area,
            "Maint_Geographical_Area": maint_geographical_area,
            "Maint_Sub_Area": f"{maint_geographical_area[0]}{maint_area}",
            "Maint_ASM": "[MAINT_ASM_NAME_PLACEHOLDER]",
            "Maint_FSM": "[MAINT_FSM_NAME_PLACEHOLDER]",
            "Maint_Car_Wash_Tech": "[TECH_NAME_PLACEHOLDER]",
            "Maint_Equipment_Tech": "[TECH_NAME_PLACEHOLDER]",
            "Maint_Facility_Support_Tech": "[TECH_NAME_PLACEHOLDER]",
            "BMS_Status": random.choice(["ACTIVE", "MAINTENANCE", "UPGRADE_NEEDED"]),
            "Construction_Maintenance": random.choice(["NONE", "MINOR", "MAJOR", "PLANNED"]),
            "HFTD_System": random.choice(["STANDARD", "ENHANCED", "PREMIUM"]),
            "Private_Systems": random.choice(["NONE", "PARTIAL", "FULL"])
        }

    def _generate_store_identifiers(self, store_number: int) -> Dict:
        """Generate various store identification numbers and related fields"""
        return {
            "FEIN": f"{random.randint(10,99)}-{random.randint(1000000,9999999)}",
            "Fuelman_Site_ID": random.randint(100000, 999999),
            "PSDCode": random.randint(1000, 9999),
            "Store_Kronos_ID": random.randint(10000, 99999),
            "Store_Name": f"SHEETZ #{store_number}",
            "Store_Number": store_number,
            "Store_Number_Label": f"Store #{store_number}",
            "Store_Folder_Redirect": f"\\\\sheetz\\stores\\{store_number}",
            "Store_Airport_Lookup": None  # Special case for airport locations
        }

    def _generate_contact_info(self) -> Dict:
        """Generate contact information fields"""
        return {
            "Telephone_Number": f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            "Fax_Telephone_Number": f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            "Rollover_Phone_Line": f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            "Special_Directions": "[SPECIAL_DIRECTIONS_PLACEHOLDER]"
        }

    def _generate_operations_info(self) -> Dict:
        """Generate operations related information"""
        return {
            "RPOS_Major_Revision": round(random.uniform(2.0, 4.0), 1),
            "SS_Store_Config_Type": random.randint(1, 8),
            "Hours_Closed": random.choice([
                "NONE - 24/7",
                "12AM-4AM",
                "1AM-4AM",
                "2AM-4AM"
            ]),
            "Drive_Thru": random.random() < 0.7,
            "Drive_Thru_POS_Number": random.randint(1, 4),
            "Main_Register": random.randint(1, 4),
            "Going_Home_Work": random.choice(["HOME", "WORK", "MIXED"]),
            "Community": random.choice([True, False]),
            "School": random.random() < 0.3,
            "Kirk_Key": random.choice([True, False]),
            "Smart_Safe": True,  # Modern standard
            "ATM": random.choice(["Y", "N"]),
            "Sells_Alcohol": random.choice([True, False]),
            "Alcohol_Type": random.choice(["NONE", "BEER", "BEER_WINE", "FULL"]),
            "Beer_Wine": random.choice([True, False]),
            "VA_ABC_DELIVERY_PERMIT": random.choice([None, "PENDING", "ACTIVE"])
        }

    def generate_store(self) -> Dict:
        """Generate a complete store record with all fields"""
        # Generate core identifiers and profile
        store_number, store_type, profile = self._generate_core_identifiers()
        
        # Generate physical specifications
        physical_specs = self._generate_physical_specs(store_type, profile)
        
        # Build complete store record
        store_record = {
            **self._generate_store_identifiers(store_number),
            **self._generate_location_fields(store_type, profile),
            **physical_specs,
            **self._generate_fuel_services(store_type, profile),
            **self._generate_ev_charging(store_type, profile),
            **self._generate_employee_data(store_type, physical_specs["Store_Square_Footage"], profile),
            **self._generate_food_service(store_type, profile),
            **self._generate_digital_services(store_type, profile),
            **self._generate_car_services(store_type, profile),
            **self._generate_maintenance_info(store_type),
            **self._generate_contact_info(),
            **self._generate_operations_info()
        }
        
        return store_record

    def generate_batch(self, size: int) -> List[Dict]:
        """Generate a batch of store records"""
        return [self.generate_store() for _ in range(size)]