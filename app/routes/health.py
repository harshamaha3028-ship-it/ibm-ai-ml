from fastapi import APIRouter
from app.utils.model_loader import model_loader
from app.database import is_using_mysql

router = APIRouter(tags=["Health"])

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "service": "Automobile Efficiency Regressor & Vehicle Class Profiler",
        "models_loaded": model_loader.is_loaded,
        "database_backend": "MySQL" if is_using_mysql() else "SQLite (Fallback)",
        "version": "1.0.0"
    }
