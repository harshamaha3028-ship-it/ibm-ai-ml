import pytest
from app.schemas.vehicle import VehicleInput
from app.services.profiling_service import profiling_service
from app.utils.model_loader import model_loader

def test_profiling_service_lightweight():
    vehicle = VehicleInput(
        cylinders=4,
        displacement=90.0,
        horsepower=65.0,
        weight=1950.0,
        acceleration=18.0,
        model_year=80,
        origin=3
    )
    profile = profiling_service.profile_vehicle(vehicle, estimated_mpg=33.5)

    assert isinstance(profile.cluster_id, int)
    assert isinstance(profile.profile_name, str)
    assert len(profile.pca_coordinates) == 2
    assert profile.distance_to_centroid >= 0.0
    assert profile.characteristics.average_mpg > 0.0

def test_profiling_service_heavy_v8():
    heavy_v8 = VehicleInput(
        cylinders=8,
        displacement=400.0,
        horsepower=200.0,
        weight=4500.0,
        acceleration=11.5,
        model_year=71,
        origin=1
    )
    profile = profiling_service.profile_vehicle(heavy_v8, estimated_mpg=13.0)

    assert isinstance(profile.cluster_id, int)
    assert "Heavy" in profile.profile_name or "Power" in profile.profile_name or "Cruiser" in profile.profile_name
    assert profile.characteristics.average_weight > 3000.0
