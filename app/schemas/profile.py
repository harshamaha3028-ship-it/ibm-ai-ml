from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ProfileCharacteristics(BaseModel):
    average_mpg: float
    average_weight: float
    average_horsepower: float
    average_displacement: float
    average_cylinders: float
    average_acceleration: Optional[float] = None
    sample_count: Optional[int] = None
    percentage_of_fleet: Optional[float] = None

class ProfileResponse(BaseModel):
    cluster_id: int
    profile_name: str
    characteristics: ProfileCharacteristics
    key_characteristics: List[str]
    pca_coordinates: List[float]
    distance_to_centroid: float

class ClusterSummaryItem(BaseModel):
    cluster_id: int
    profile_name: str
    characteristics: ProfileCharacteristics
    key_characteristics: List[str]
    pca_centroid: List[float]

class ClusterListResponse(BaseModel):
    num_clusters: int
    features_used: List[str]
    pca_explained_variance_ratio: List[float]
    clusters: List[ClusterSummaryItem]
