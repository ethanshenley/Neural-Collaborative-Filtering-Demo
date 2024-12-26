# src/inference/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ProductMetadata(BaseModel):
    """Product metadata model"""
    total_purchases: int
    unique_customers: int
    common_pairs: List[Dict[str, Any]]
    hourly_pattern: List[float]
    daily_pattern: List[float]

class ProductRecommendation(BaseModel):
    """Single product recommendation"""
    product_id: str
    name: str
    category_id: str
    department_id: str
    score: float = Field(..., description="Recommendation score between 0 and 1")
    price: float
    loyalty_score: float
    metadata: ProductMetadata

class RecommendationRequest(BaseModel):
    """Recommendation request parameters"""
    customer_id: str
    num_recommendations: int = Field(default=10, ge=1, le=100)
    category_filter: Optional[str] = None
    include_features: bool = False

class RecommendationMetadata(BaseModel):
    """Recommendation response metadata"""
    inference_time_ms: float
    model_version: str
    features_used: Optional[List[str]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RecommendationResponse(BaseModel):
    """Complete recommendation response"""
    recommendations: List[ProductRecommendation]
    metadata: RecommendationMetadata

class BatchRecommendationRequest(BaseModel):
    """Batch recommendation request"""
    customer_ids: List[str] = Field(..., max_items=100)
    num_recommendations: int = Field(default=10, ge=1, le=100)
    category_filter: Optional[str] = None

class BatchRecommendationResponse(BaseModel):
    """Batch recommendation response"""
    results: Dict[str, List[ProductRecommendation]]
    metadata: RecommendationMetadata

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)