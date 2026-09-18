from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.vehicle import VehicleInput
from app.schemas.prediction import (
    PredictionResponse, AnalysisResponse,
    PredictionHistoryResponse, ModelInformationResponse
)
from app.services.prediction_service import prediction_service
from app.services.analytics_service import analytics_service
from app.database import get_predictions_history, delete_all_predictions
from app.utils.model_loader import model_loader

router = APIRouter(tags=["Prediction & Analysis"])

@router.post("/predict", response_model=PredictionResponse)
def predict_mpg(vehicle: VehicleInput):
    """
    Predict vehicle Miles Per Gallon (MPG) using regularized regression.
    """
    try:
        predicted_mpg, efficiency_category, contributions = prediction_service.predict_mpg(vehicle)
        meta = model_loader.metadata
        return PredictionResponse(
            predicted_mpg=predicted_mpg,
            efficiency_category=efficiency_category,
            model_name=meta.get("best_regression_model", "Linear Regression"),
            feature_contributions=contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/analyze", response_model=AnalysisResponse)
def analyze_vehicle(vehicle: VehicleInput):
    """
    Unified endpoint: Predicts MPG, groups vehicle into physical/performance cluster profile,
    computes explainability breakdown, and saves analysis to database.
    """
    try:
        return analytics_service.analyze_vehicle(vehicle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@router.get("/predictions", response_model=PredictionHistoryResponse)
def get_history(
    search: Optional[str] = Query(None, description="Search profile name, origin, or efficiency"),
    efficiency: Optional[str] = Query(None, description="Filter by efficiency category"),
    profile: Optional[str] = Query(None, description="Filter by vehicle profile"),
    sort_by: str = Query("id", description="Column to sort by"),
    order: str = Query("desc", description="Sort direction ('asc' or 'desc')"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(15, ge=1, le=100, description="Items per page")
):
    """
    Retrieve paginated prediction logs with filtering and sorting.
    """
    try:
        result = get_predictions_history(
            search=search,
            efficiency=efficiency,
            profile=profile,
            sort_by=sort_by,
            order=order,
            page=page,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

@router.delete("/predictions")
def clear_history():
    """
    Clear all stored predictions from the database.
    """
    try:
        delete_all_predictions()
        return {"status": "success", "message": "Prediction history cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/model-info", response_model=ModelInformationResponse)
def get_model_info():
    """
    Returns training metrics comparison across Linear, Ridge, Lasso, and ElasticNet models,
    standardized coefficients, and dataset thresholds.
    """
    meta = model_loader.metadata
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata not loaded.")
    return ModelInformationResponse(
        best_model=meta.get("best_regression_model", "Linear Regression"),
        models_comparison=meta.get("model_comparison", {}),
        feature_importances=meta.get("feature_impacts", []),
        efficiency_thresholds=meta.get("efficiency_thresholds", {}),
        dataset_statistics=meta.get("dataset_statistics", {})
    )
