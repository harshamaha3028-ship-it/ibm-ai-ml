import pytest
import numpy as np
import pandas as pd
from app.utils.preprocessing import AutoMPGFeatureEngineer
from app.schemas.vehicle import VehicleInput
from app.services.prediction_service import prediction_service
from app.utils.model_loader import model_loader

def test_feature_engineering_shape_and_columns():
    fe = AutoMPGFeatureEngineer(reference_year=82)
    sample_df = pd.DataFrame([{
        'cylinders': 4,
        'displacement': 120.0,
        'horsepower': 85.0,
        'weight': 2300.0,
        'acceleration': 15.5,
        'model_year': 78,
        'origin': 1
    }])
    transformed = fe.transform(sample_df)
    
    assert transformed.shape[0] == 1
    assert 'power_to_weight' in transformed.columns
    assert 'displacement_per_cylinder' in transformed.columns
    assert 'weight_per_cylinder' in transformed.columns
    assert 'vehicle_age' in transformed.columns
    assert 'origin_usa' in transformed.columns
    assert 'origin_europe' in transformed.columns
    assert 'origin_japan' in transformed.columns

def test_prediction_service_valid_vehicle():
    vehicle = VehicleInput(
        cylinders=4,
        displacement=100.0,
        horsepower=75.0,
        weight=2150.0,
        acceleration=16.0,
        model_year=80,
        origin=3,
        car_name="Test Japanese Hatchback"
    )
    predicted_mpg, category, contributions = prediction_service.predict_mpg(vehicle)

    assert isinstance(predicted_mpg, float)
    assert predicted_mpg > 15.0 # Lightweight 4-cyl should have high efficiency
    assert category in ["High Efficiency", "Moderate Efficiency", "Low Efficiency"]
    assert len(contributions) > 0
    assert any(c.feature == 'weight' for c in contributions)

def test_prediction_heavy_v8_low_efficiency():
    heavy_vehicle = VehicleInput(
        cylinders=8,
        displacement=380.0,
        horsepower=180.0,
        weight=4400.0,
        acceleration=11.0,
        model_year=72,
        origin=1,
        car_name="Test Heavy Muscle V8"
    )
    predicted_mpg, category, contributions = prediction_service.predict_mpg(heavy_vehicle)

    assert predicted_mpg < 20.0
    assert category == "Low Efficiency"
