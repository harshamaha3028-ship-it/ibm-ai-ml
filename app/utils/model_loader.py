import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import settings

class ModelLoader:
    _instance: Optional['ModelLoader'] = None

    def __init__(self):
        self.regression_model = None
        self.preprocessing_pipeline = None
        self.clustering_model = None
        self.cluster_scaler = None
        self.pca_model = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False

    @classmethod
    def get_instance(cls) -> 'ModelLoader':
        if cls._instance is None:
            cls._instance = ModelLoader()
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        models_dir = settings.MODELS_DIR

        # Load Metadata
        meta_path = models_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print(f"[ModelLoader Warning] Metadata not found at {meta_path}. Run training scripts first.")

        # Load Regression Model
        reg_path = models_dir / "regression_model.joblib"
        if reg_path.exists():
            self.regression_model = joblib.load(reg_path)
        else:
            print(f"[ModelLoader Warning] Regression model not found at {reg_path}.")

        # Load Preprocessing Pipeline
        prep_path = models_dir / "preprocessing_pipeline.joblib"
        if prep_path.exists():
            self.preprocessing_pipeline = joblib.load(prep_path)
        else:
            print(f"[ModelLoader Warning] Preprocessing pipeline not found at {prep_path}.")

        # Load Clustering Model
        clust_path = models_dir / "clustering_model.joblib"
        if clust_path.exists():
            self.clustering_model = joblib.load(clust_path)
        else:
            print(f"[ModelLoader Warning] Clustering model not found at {clust_path}.")

        # Load Cluster Scaler
        scaler_path = models_dir / "cluster_scaler.joblib"
        if scaler_path.exists():
            self.cluster_scaler = joblib.load(scaler_path)
        else:
            print(f"[ModelLoader Warning] Cluster scaler not found at {scaler_path}.")

        # Load PCA
        pca_path = models_dir / "pca.joblib"
        if pca_path.exists():
            self.pca_model = joblib.load(pca_path)
        else:
            print(f"[ModelLoader Warning] PCA model not found at {pca_path}.")

        self.is_loaded = True
        print("[ModelLoader] Successfully loaded all ML artifacts and metadata.")

model_loader = ModelLoader.get_instance()
