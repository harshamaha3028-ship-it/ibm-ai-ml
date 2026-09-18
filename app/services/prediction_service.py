import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from app.utils.model_loader import model_loader
from app.schemas.vehicle import VehicleInput
from app.schemas.prediction import FeatureContribution, PredictionResponse
from app.utils.preprocessing import ORIGIN_MAP

class PredictionService:
    def __init__(self):
        self.loader = model_loader

    def predict_mpg(self, vehicle: VehicleInput) -> Tuple[float, str, List[FeatureContribution]]:
        """
        Runs regression inference and generates coefficient-weighted feature attribution.
        """
        # Build raw DataFrame for preprocessing pipeline
        df_raw = pd.DataFrame([{
            'cylinders': float(vehicle.cylinders),
            'displacement': float(vehicle.displacement),
            'horsepower': float(vehicle.horsepower),
            'weight': float(vehicle.weight),
            'acceleration': float(vehicle.acceleration),
            'model_year': float(vehicle.model_year),
            'origin': int(vehicle.origin)
        }])

        # Transform using fitted pipeline
        X_processed = self.loader.preprocessing_pipeline.transform(df_raw)
        
        # Predict MPG
        predicted_mpg_raw = float(self.loader.regression_model.predict(X_processed)[0])
        predicted_mpg = max(5.0, round(predicted_mpg_raw, 2))

        # Determine Efficiency Category from metadata percentile thresholds
        meta = self.loader.metadata
        thresholds = meta.get('efficiency_thresholds', {})
        low_th = thresholds.get('low_threshold', 19.0)
        high_th = thresholds.get('high_threshold', 27.0)

        if predicted_mpg <= low_th:
            efficiency_category = "Low Efficiency"
        elif predicted_mpg <= high_th:
            efficiency_category = "Moderate Efficiency"
        else:
            efficiency_category = "High Efficiency"

        # Explainability Engine:
        # Calculate feature contributions = (scaled_value * coefficient)
        feature_names = meta.get('feature_names', [])
        coefficients = self.loader.regression_model.coef_
        scaled_values = X_processed.values[0] if hasattr(X_processed, 'values') else X_processed[0]
        
        contributions: List[FeatureContribution] = []
        for name, scaled_val, coef in zip(feature_names, scaled_values, coefficients):
            impact_score = float(scaled_val * coef)
            direction = "Increases MPG" if impact_score > 0 else ("Decreases MPG" if impact_score < 0 else "Neutral")
            
            # Formulate user-friendly explanation
            desc = ""
            if name == 'weight':
                desc = f"Vehicle curb weight ({vehicle.weight} lbs) {'increases inertia & fuel load' if vehicle.weight > 3000 else 'is relatively light, saving fuel'}."
            elif name == 'horsepower':
                desc = f"Horsepower rating ({vehicle.horsepower} hp) {'demands more fuel per combustion cycle' if vehicle.horsepower > 100 else 'conserves fuel delivery'}."
            elif name == 'displacement':
                desc = f"Engine displacement ({vehicle.displacement} cu.in.) capacity."
            elif name == 'cylinders':
                desc = f"{vehicle.cylinders}-cylinder engine configuration."
            elif name == 'model_year':
                desc = f"Model year 19{vehicle.model_year} incorporates aerodynamic/emissions evolutions."
            elif name == 'acceleration':
                desc = f"0-60 acceleration ({vehicle.acceleration}s) performance profile."
            elif name == 'power_to_weight':
                desc = f"Power-to-weight ratio ({(vehicle.horsepower/vehicle.weight):.4f} hp/lb)."
            elif name == 'origin_usa':
                desc = f"North American manufacturing specification standard."
            elif name == 'origin_japan':
                desc = f"Japanese compact manufacturing engineering."
            elif name == 'origin_europe':
                desc = f"European precision tuning and gearing."
            else:
                desc = f"Derived characteristic: {name}."

            contributions.append(FeatureContribution(
                feature=str(name),
                raw_value=getattr(vehicle, name, round(float(scaled_val), 2)),
                coefficient=round(float(coef), 4),
                impact_direction=direction,
                description=desc
            ))

        # Sort contributions by absolute impact
        contributions.sort(key=lambda x: abs(x.coefficient), reverse=True)

        return predicted_mpg, efficiency_category, contributions

prediction_service = PredictionService()
