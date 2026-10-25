import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from app.utils.model_loader import model_loader
from app.schemas.vehicle import VehicleInput
from app.schemas.profile import ProfileResponse, ProfileCharacteristics

class ProfilingService:
    def __init__(self):
        self.loader = model_loader

    def profile_vehicle(self, vehicle: VehicleInput, estimated_mpg: float) -> ProfileResponse:
        """
        Assigns vehicle to a physical/performance cluster and projects it into 2D PCA space.
        """
        # Features used in clustering
        feat_df = pd.DataFrame([{
            'cylinders': float(vehicle.cylinders),
            'displacement': float(vehicle.displacement),
            'horsepower': float(vehicle.horsepower),
            'weight': float(vehicle.weight),
            'acceleration': float(vehicle.acceleration),
            'mpg': float(estimated_mpg)
        }])

        # Scale features
        scaled_feat = self.loader.cluster_scaler.transform(feat_df)

        # Predict cluster ID
        cluster_id = int(self.loader.clustering_model.predict(scaled_feat)[0])

        # PCA 2D coordinates
        pca_coords = self.loader.pca_model.transform(scaled_feat)[0]
        pca_x = round(float(pca_coords[0]), 3)
        pca_y = round(float(pca_coords[1]), 3)

        # Distance to centroid in scaled feature space
        centroid = self.loader.clustering_model.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(scaled_feat[0] - centroid))

        # Retrieve profile metadata
        meta = self.loader.metadata.get('clustering', {})
        cluster_profiles = meta.get('cluster_profiles', {})
        profile_data = cluster_profiles.get(str(cluster_id), {})

        profile_name = profile_data.get('profile_name', f"Cluster {cluster_id}")
        key_chars = profile_data.get('key_characteristics', [])

        characteristics = ProfileCharacteristics(
            average_mpg=profile_data.get('average_mpg', estimated_mpg),
            average_weight=profile_data.get('average_weight', vehicle.weight),
            average_horsepower=profile_data.get('average_horsepower', vehicle.horsepower),
            average_displacement=profile_data.get('average_displacement', vehicle.displacement),
            average_cylinders=profile_data.get('average_cylinders', vehicle.cylinders),
            average_acceleration=profile_data.get('average_acceleration', vehicle.acceleration),
            sample_count=profile_data.get('sample_count', 0),
            percentage_of_fleet=profile_data.get('percentage_of_fleet', 0.0)
        )

        return ProfileResponse(
            cluster_id=cluster_id,
            profile_name=profile_name,
            characteristics=characteristics,
            key_characteristics=key_chars,
            pca_coordinates=[pca_x, pca_y],
            distance_to_centroid=round(distance, 3)
        )

profiling_service = ProfilingService()
