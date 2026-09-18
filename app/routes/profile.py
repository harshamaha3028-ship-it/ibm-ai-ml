from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.vehicle import VehicleInput
from app.schemas.profile import ProfileResponse, ClusterListResponse, ClusterSummaryItem, ProfileCharacteristics
from app.services.profiling_service import profiling_service
from app.services.prediction_service import prediction_service
from app.utils.model_loader import model_loader

router = APIRouter(tags=["Vehicle Profiling & Clustering"])

@router.post("/profile", response_model=ProfileResponse)
def get_vehicle_profile(vehicle: VehicleInput):
    """
    Classify vehicle into unsupervised cluster archetype based on physical & performance characteristics.
    """
    try:
        # If MPG isn't provided directly, infer it through the regression model
        predicted_mpg, _, _ = prediction_service.predict_mpg(vehicle)
        return profiling_service.profile_vehicle(vehicle, predicted_mpg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")

@router.get("/clusters", response_model=ClusterListResponse)
def get_clusters():
    """
    Retrieve fleet clustering archetypes, centroids, PCA variance, and characteristics.
    """
    meta = model_loader.metadata.get("clustering", {})
    if not meta:
        raise HTTPException(status_code=404, detail="Clustering metadata not available.")

    profiles_dict = meta.get("cluster_profiles", {})
    cluster_items: List[ClusterSummaryItem] = []

    for c_id, prof in profiles_dict.items():
        char = ProfileCharacteristics(
            average_mpg=prof.get("average_mpg", 0.0),
            average_weight=prof.get("average_weight", 0.0),
            average_horsepower=prof.get("average_horsepower", 0.0),
            average_displacement=prof.get("average_displacement", 0.0),
            average_cylinders=prof.get("average_cylinders", 0.0),
            average_acceleration=prof.get("average_acceleration", 0.0),
            sample_count=prof.get("sample_count", 0),
            percentage_of_fleet=prof.get("percentage_of_fleet", 0.0)
        )
        cluster_items.append(ClusterSummaryItem(
            cluster_id=int(prof.get("cluster_id", c_id)),
            profile_name=prof.get("profile_name", f"Cluster {c_id}"),
            characteristics=char,
            key_characteristics=prof.get("key_characteristics", []),
            pca_centroid=prof.get("pca_centroid", [0.0, 0.0])
        ))

    return ClusterListResponse(
        num_clusters=meta.get("num_clusters", len(cluster_items)),
        features_used=meta.get("features_used", []),
        pca_explained_variance_ratio=meta.get("pca_explained_variance_ratio", []),
        clusters=cluster_items
    )
