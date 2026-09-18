import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Mapping for origin
ORIGIN_MAP = {
    1: 'USA',
    2: 'Europe',
    3: 'Japan'
}

ORIGIN_NAME_TO_CODE = {
    'USA': 1,
    'Europe': 2,
    'Japan': 3,
    'usa': 1,
    'europe': 2,
    'japan': 3
}

REFERENCE_YEAR = 82

class AutoMPGFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer that engineers domain-specific automobile features:
    - power_to_weight: horsepower / weight
    - displacement_per_cylinder: displacement / cylinders
    - weight_per_cylinder: weight / cylinders
    - vehicle_age: REFERENCE_YEAR - model_year
    """
    def __init__(self, reference_year: int = REFERENCE_YEAR):
        self.reference_year = reference_year
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Support both DataFrame and NumPy array
        if isinstance(X, np.ndarray):
            # Assumed column order: cylinders, displacement, horsepower, weight, acceleration, model_year, origin
            df = pd.DataFrame(X, columns=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin'])
        else:
            df = X.copy()

        df_out = pd.DataFrame(index=df.index)
        
        # Base numerical features
        df_out['cylinders'] = df['cylinders'].astype(float)
        df_out['displacement'] = df['displacement'].astype(float)
        df_out['horsepower'] = df['horsepower'].astype(float)
        df_out['weight'] = df['weight'].astype(float)
        df_out['acceleration'] = df['acceleration'].astype(float)
        df_out['model_year'] = df['model_year'].astype(float)
        
        # Engineered features
        # Avoid division by zero with small epsilon
        eps = 1e-6
        df_out['power_to_weight'] = df_out['horsepower'] / (df_out['weight'] + eps)
        df_out['displacement_per_cylinder'] = df_out['displacement'] / (df_out['cylinders'] + eps)
        df_out['weight_per_cylinder'] = df_out['weight'] / (df_out['cylinders'] + eps)
        df_out['vehicle_age'] = self.reference_year - df_out['model_year']
        
        # One-hot encoded origin columns
        # origin: 1 (USA), 2 (Europe), 3 (Japan)
        origin_series = df['origin'].apply(lambda v: ORIGIN_NAME_TO_CODE.get(str(v).strip().lower(), v) if isinstance(v, str) else v)
        df_out['origin_usa'] = (origin_series == 1).astype(float)
        df_out['origin_europe'] = (origin_series == 2).astype(float)
        df_out['origin_japan'] = (origin_series == 3).astype(float)
        
        self.feature_names_out_ = list(df_out.columns)
        return df_out

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_)
        return np.array([
            'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year',
            'power_to_weight', 'displacement_per_cylinder', 'weight_per_cylinder', 'vehicle_age',
            'origin_usa', 'origin_europe', 'origin_japan'
        ])
