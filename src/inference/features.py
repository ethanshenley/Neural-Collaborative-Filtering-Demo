# src/inference/features.py

from google.cloud import bigquery
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import yaml
from datetime import datetime, timezone
import torch

class FeatureProcessor:
    """Process and engineer features for inference"""
    
    def __init__(self):
        # Load config
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
            self.feature_config = config["inference"]["feature_engineering"]
            
        self.bq_client = bigquery.Client()
        self.max_sequence_length = self.feature_config["max_sequence_length"]
        
    async def get_features(self, customer_id: str) -> Dict[str, Any]:
        """Get engineered features for a customer"""
        try:
            # Get raw features from BigQuery
            query = self._build_feature_query(customer_id)
            df = await self.bq_client.query(query).result().to_dataframe_async()
            
            if df.empty:
                raise ValueError(f"No data found for customer {customer_id}")
            
            # Process features
            features = self._process_features(df)
            
            # Add temporal features
            features.update(
                self._add_temporal_features(df)
            )
            
            # Add sequence features
            features.update(
                self._add_sequence_features(df)
            )
            
            return features
            
        except Exception as e:
            logging.error(f"Feature processing error for {customer_id}: {str(e)}")
            raise
            
    def _build_feature_query(self, customer_id: str) -> str:
        """Build BigQuery feature extraction query"""
        return f"""
        WITH user_features AS (
            SELECT *
            FROM `user_features_enriched`
            WHERE cardnumber = @customer_id
        ),
        recent_transactions AS (
            SELECT 
                thf.physical_date_time,
                tbf.inventory_code as product_id,
                tbf.category_id,
                tbf.department_id,
                tbf.extended_retail as amount
            FROM `transaction_header_fact` thf
            JOIN `transaction_body_fact` tbf 
                ON thf.store_number = tbf.store_number
                AND thf.transaction_number = tbf.transaction_number
            WHERE thf.cust_code = @customer_id
            ORDER BY thf.physical_date_time DESC
            LIMIT {self.max_sequence_length}
        )
        SELECT 
            uf.*,
            ARRAY_AGG(STRUCT(
                rt.physical_date_time,
                rt.product_id,
                rt.category_id,
                rt.department_id,
                rt.amount
            )) as recent_transactions
        FROM user_features uf
        LEFT JOIN recent_transactions rt
        GROUP BY uf.*
        """
        
    def _process_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process basic user features"""
        row = df.iloc[0]
        
        features = {
            "user_id": row["cardnumber"],
            "age": row["age"],
            "gender": row["gender"],
            "enrollment_status": row["enrollment_status"],
            "lifetime_points": row["lifetime_points"],
            "dollars_redeemed": row["dollars_redeemed"],
            "account_age_days": row["account_age_days"],
            "interaction_frequency": row["interaction_frequency"],
            "points_per_interaction": row["points_per_interaction"]
        }
        
        # Add categorical features
        for cat_feature in self.feature_config["categorical_features"]:
            if cat_feature in row:
                features[cat_feature] = row[cat_feature]
                
        return features
        
    def _add_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Add temporal features"""
        now = datetime.now(timezone.utc)
        row = df.iloc[0]
        
        temporal_features = {}
        for feature in self.feature_config["temporal_features"]:
            if feature == "hour_of_day":
                temporal_features["hour"] = now.hour
            elif feature == "day_of_week":
                temporal_features["day"] = now.weekday()
            elif feature == "days_since_last_purchase":
                if pd.notnull(row["last_purchase_date"]):
                    last_purchase = pd.to_datetime(row["last_purchase_date"])
                    temporal_features["days_since"] = (
                        now - last_purchase
                    ).days
                else:
                    temporal_features["days_since"] = 0
                    
        return temporal_features
        
    def _add_sequence_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process sequence features from recent transactions"""
        row = df.iloc[0]
        recent_transactions = row["recent_transactions"]
        
        if not recent_transactions:
            return {"sequence": []}
            
        # Sort by timestamp and take most recent
        transactions = sorted(
            recent_transactions,
            key=lambda x: x["physical_date_time"],
            reverse=True
        )[:self.max_sequence_length]
        
        # Extract product IDs
        sequence = [t["product_id"] for t in transactions]
        
        # Pad sequence if needed
        if len(sequence) < self.max_sequence_length:
            sequence.extend(
                ["PAD"] * (self.max_sequence_length - len(sequence))
            )
            
        return {"sequence": sequence}
        
    def enrich_products(
        self,
        products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich product recommendations with metadata"""
        if not products:
            return []
            
        # Get product details from BigQuery
        product_ids = [p["product_id"] for p in products]
        query = f"""
        SELECT 
            p.*,
            STRUCT(
                common_pairs,
                hourly_sales,
                daily_sales,
                purchase_loyalty_score,
                unique_customers,
                total_purchases,
                total_revenue,
                avg_price,
                avg_quantity
            ) as metrics
        FROM `product_features_enriched` p
        WHERE product_id IN UNNEST(@product_ids)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "product_ids", 
                    "STRING", 
                    product_ids
                )
            ]
        )
        
        try:
            df = self.bq_client.query(query, job_config=job_config).to_dataframe()
            
            # Create lookup dictionary
            product_details = {}
            for _, row in df.iterrows():
                product_details[row["product_id"]] = {
                    "name": row["product_name"],
                    "category_id": row["category_id"],
                    "department_id": row["department_id"],
                    "metrics": {
                        "purchase_loyalty_score": float(row["metrics"]["purchase_loyalty_score"]),
                        "unique_customers": int(row["metrics"]["unique_customers"]),
                        "total_purchases": int(row["metrics"]["total_purchases"]),
                        "avg_price": float(row["metrics"]["avg_price"]),
                        "popularity_score": self._calculate_popularity_score(row)
                    },
                    "temporal_patterns": {
                        "hourly": row["metrics"]["hourly_sales"],
                        "daily": row["metrics"]["daily_sales"]
                    },
                    "common_pairs": row["metrics"]["common_pairs"][:5]  # Top 5 related products
                }
            
            # Enrich recommendations
            enriched_products = []
            for product in products:
                product_id = product["product_id"]
                if product_id in product_details:
                    enriched = {
                        **product,
                        **product_details[product_id],
                        "explanation": self._generate_explanation(
                            product, 
                            product_details[product_id]
                        )
                    }
                    enriched_products.append(enriched)
            
            return enriched_products
            
        except Exception as e:
            logging.error(f"Product enrichment error: {str(e)}")
            return products

    def _calculate_popularity_score(self, row: pd.Series) -> float:
        """Calculate normalized popularity score"""
        metrics = row["metrics"]
        
        # Combine multiple signals
        signals = {
            "purchase_loyalty": metrics["purchase_loyalty_score"],
            "unique_customers": np.log1p(metrics["unique_customers"]),
            "total_purchases": np.log1p(metrics["total_purchases"])
        }
        
        # Normalize each signal to [0, 1]
        normalized = {}
        for key, value in signals.items():
            min_val = self.metric_ranges[key]["min"]
            max_val = self.metric_ranges[key]["max"]
            normalized[key] = (value - min_val) / (max_val - min_val)
            
        # Weighted combination
        weights = {
            "purchase_loyalty": 0.4,
            "unique_customers": 0.3,
            "total_purchases": 0.3
        }
        
        score = sum(
            normalized[key] * weights[key] 
            for key in normalized
        )
        
        return float(score)

    def _generate_explanation(
        self, 
        recommendation: Dict[str, Any],
        details: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation for recommendation"""
        score = recommendation.get("score", 0)
        metrics = details["metrics"]
        
        explanations = []
        
        # Base explanation on score
        if score > 0.9:
            explanations.append("Highly recommended based on your preferences")
        elif score > 0.7:
            explanations.append("Good match for your tastes")
            
        # Add popularity context
        if metrics["unique_customers"] > 1000:
            explanations.append(
                f"Popular choice with {metrics['unique_customers']} customers"
            )
            
        # Add loyalty context
        if metrics["purchase_loyalty_score"] > 0.7:
            explanations.append("Customers frequently return for this item")
            
        # Add price context if available
        if "avg_price" in metrics:
            explanations.append(f"Average price: ${metrics['avg_price']:.2f}")
            
        # Combine explanations
        return " • ".join(explanations)

    def update_metric_ranges(self):
        """Update metric ranges for normalization"""
        query = """
        SELECT
            MIN(purchase_loyalty_score) as min_loyalty,
            MAX(purchase_loyalty_score) as max_loyalty,
            MIN(unique_customers) as min_customers,
            MAX(unique_customers) as max_customers,
            MIN(total_purchases) as min_purchases,
            MAX(total_purchases) as max_purchases
        FROM `product_features_enriched`
        """
        
        df = self.bq_client.query(query).to_dataframe()
        row = df.iloc[0]
        
        self.metric_ranges = {
            "purchase_loyalty": {
                "min": float(row["min_loyalty"]),
                "max": float(row["max_loyalty"])
            },
            "unique_customers": {
                "min": np.log1p(float(row["min_customers"])),
                "max": np.log1p(float(row["max_customers"]))
            },
            "total_purchases": {
                "min": np.log1p(float(row["min_purchases"])),
                "max": np.log1p(float(row["max_purchases"]))
            }
        }

    def preprocess_features_for_model(
        self, 
        features: Dict[str, Any]
    ) -> torch.Tensor:
        """Convert features to model input format"""
        # Initialize empty tensors
        categorical_features = torch.zeros(
            len(self.feature_config["categorical_features"]),
            dtype=torch.long
        )
        numerical_features = torch.zeros(
            len(self.feature_config["numerical_features"]),
            dtype=torch.float
        )
        temporal_features = torch.zeros(
            len(self.feature_config["temporal_features"]),
            dtype=torch.float
        )
        sequence_features = torch.zeros(
            self.max_sequence_length,
            dtype=torch.long
        )
        
        # Fill categorical features
        for i, feature in enumerate(self.feature_config["categorical_features"]):
            if feature in features:
                categorical_features[i] = features[feature]
                
        # Fill numerical features
        for i, feature in enumerate(self.feature_config["numerical_features"]):
            if feature in features:
                numerical_features[i] = features[feature]
                
        # Fill temporal features
        if "temporal" in features:
            temporal = features["temporal"]
            for i, feature in enumerate(self.feature_config["temporal_features"]):
                if feature in temporal:
                    temporal_features[i] = temporal[feature]
                    
        # Fill sequence
        if "sequence" in features:
            sequence = features["sequence"]
            for i, product_id in enumerate(sequence):
                if i < self.max_sequence_length:
                    sequence_features[i] = self.product_to_idx.get(
                        product_id, 
                        0  # Use 0 for padding
                    )
                    
        return {
            "categorical": categorical_features,
            "numerical": numerical_features,
            "temporal": temporal_features,
            "sequence": sequence_features
        }