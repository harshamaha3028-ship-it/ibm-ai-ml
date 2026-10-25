from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FeatureContribution(BaseModel):
    feature: str
    raw_value: Any
    coefficient: float
    impact_direction: str # "Increases MPG", "Decreases MPG", "Neutral"
    description: str

class PredictionResponse(BaseModel):
    predicted_mpg: float = Field(..., description="Estimated Miles Per Gallon")
    efficiency_category: str = Field(..., description="Low Efficiency / Moderate Efficiency / High Efficiency")
    model_name: str
    confidence_interval: Optional[Dict[str, float]] = None
    feature_contributions: List[FeatureContribution]

class AnalysisResponse(BaseModel):
    prediction_id: Optional[int] = None
    predicted_mpg: float
    efficiency_category: str
    vehicle_profile: str
    cluster_id: int
    key_characteristics: List[str]
    feature_contributions: List[FeatureContribution]
    cluster_comparison: Dict[str, Any]
    pca_coordinates: List[float]

class PredictionHistoryItem(BaseModel):
    id: int
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: str
    predicted_mpg: float
    efficiency_category: str
    cluster_id: int
    profile_name: str
    created_at: str

class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total: int
    page: int
    limit: int
    total_pages: int

class ModelInformationResponse(BaseModel):
    best_model: str
    models_comparison: Dict[str, Any]
    feature_importances: List[Dict[str, Any]]
    efficiency_thresholds: Dict[str, Any]
    dataset_statistics: Dict[str, Any]
