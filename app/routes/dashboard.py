from fastapi import APIRouter, HTTPException
from app.services.analytics_service import analytics_service

router = APIRouter(tags=["Dashboard & Analytics"])

@router.get("/statistics")
def get_dashboard_stats():
    """
    Returns aggregated dashboard statistics, model KPI metrics, dataset distribution, and recent activity.
    """
    try:
        return analytics_service.get_dashboard_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard statistics: {str(e)}")
