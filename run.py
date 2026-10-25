import os
import sys
import uvicorn
from app.config import settings

def main():
    print("=" * 70)
    print("      AUTOMOBILE EFFICIENCY REGRESSOR & VEHICLE CLASS PROFILER")
    print("=" * 70)
    
    # Check if raw data exists, otherwise download
    data_path = settings.BASE_DIR / "data" / "raw" / "auto-mpg.csv"
    if not data_path.exists():
        print("[Auto-Setup] Dataset not found. Downloading Auto MPG dataset...")
        from training.download_data import fetch_and_save_data
        fetch_and_save_data()

    # Check if models exist, otherwise train
    model_path = settings.MODELS_DIR / "regression_model.joblib"
    if not model_path.exists():
        print("[Auto-Setup] Model artifacts not found. Training models now...")
        from training.train_regression import train_regression
        from training.train_clustering import train_clustering
        train_regression()
        train_clustering(optimal_k=3)

    print(f"\n[Server] Starting FastAPI Server on http://{settings.HOST}:{settings.PORT}")
    print(f"[Documentation] Interactive API Docs available at http://{settings.HOST}:{settings.PORT}/docs")
    print(f"[Dashboard] Web Dashboard accessible at http://{settings.HOST}:{settings.PORT}")
    print("=" * 70)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()
