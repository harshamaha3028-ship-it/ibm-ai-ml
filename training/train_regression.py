import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.preprocessing import AutoMPGFeatureEngineer, ORIGIN_MAP, REFERENCE_YEAR

# Constants & Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'auto-mpg.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'frontend', 'assets', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_regression():
    print("=== Step 1: Loading and Cleaning Data ===")
    if not os.path.exists(DATA_PATH):
        from training.download_data import fetch_and_save_data
        fetch_and_save_data()

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Features and target
    feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    target_col = 'mpg'

    # Convert horsepower to numeric
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    
    # Store clean dataset stats
    mpg_stats = {
        'min': float(df[target_col].min()),
        'max': float(df[target_col].max()),
        'mean': float(df[target_col].mean()),
        'median': float(df[target_col].median()),
        'std': float(df[target_col].std()),
        'count': int(len(df))
    }

    X = df[feature_cols]
    y = df[target_col]

    print("\n=== Step 2: Leakage-Free Train/Test Split (80/20) ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Calculate percentile-based efficiency thresholds on training target only
    low_thresh = float(np.percentile(y_train, 33.33))
    high_thresh = float(np.percentile(y_train, 66.67))
    print(f"Dataset-derived MPG Efficiency Thresholds (Train Set):")
    print(f"  - Low Efficiency:      <= {low_thresh:.1f} MPG")
    print(f"  - Moderate Efficiency:  {low_thresh:.1f} - {high_thresh:.1f} MPG")
    print(f"  - High Efficiency:      > {high_thresh:.1f} MPG")

    print("\n=== Step 3: Building Preprocessing Pipeline ===")
    # Raw imputer for initial features
    raw_imputer = SimpleImputer(strategy='median')
    feature_engineer = AutoMPGFeatureEngineer(reference_year=REFERENCE_YEAR)
    scaler = StandardScaler()

    # Create full preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', raw_imputer),
        ('feature_engineer', feature_engineer),
        ('scaler', scaler)
    ])

    # Fit preprocessing strictly on training data
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    engineered_feature_names = feature_engineer.get_feature_names_out()
    print(f"Engineered features ({len(engineered_feature_names)}): {list(engineered_feature_names)}")

    print("\n=== Step 4: Cross-Validation & Hyperparameter Tuning ===")
    alphas = np.logspace(-3, 3, 50)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Candidate models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': RidgeCV(alphas=alphas, cv=cv),
        'Lasso Regression': LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=5000),
        'ElasticNet Regression': ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=cv, random_state=42, max_iter=5000)
    }

    comparison_results = {}
    fitted_models = {}

    for name, model in models.items():
        # Fit model on processed training data
        model.fit(X_train_processed, y_train)
        fitted_models[name] = model

        # 5-fold cross-validation scores
        cv_scores = cross_validate(
            model, X_train_processed, y_train, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'neg_mean_squared_error']
        )

        # Test set evaluation
        y_pred_test = model.predict(X_test_processed)
        y_pred_train = model.predict(X_train_processed)

        test_r2 = float(r2_score(y_test, y_pred_test))
        test_mae = float(mean_absolute_error(y_test, y_pred_test))
        test_mse = float(mean_squared_error(y_test, y_pred_test))
        test_rmse = float(np.sqrt(test_mse))

        train_r2 = float(r2_score(y_train, y_pred_train))
        cv_r2_mean = float(cv_scores['test_r2'].mean())
        cv_r2_std = float(cv_scores['test_r2'].std())
        cv_mae_mean = float(-cv_scores['test_neg_mean_absolute_error'].mean())
        cv_rmse_mean = float(-cv_scores['test_neg_root_mean_squared_error'].mean())

        best_params = {}
        if hasattr(model, 'alpha_'):
            best_params['alpha'] = float(model.alpha_)
        if hasattr(model, 'l1_ratio_'):
            best_params['l1_ratio'] = float(model.l1_ratio_)

        comparison_results[name] = {
            'train_r2': round(train_r2, 4),
            'cv_r2_mean': round(cv_r2_mean, 4),
            'cv_r2_std': round(cv_r2_std, 4),
            'cv_mae_mean': round(cv_mae_mean, 4),
            'cv_rmse_mean': round(cv_rmse_mean, 4),
            'test_r2': round(test_r2, 4),
            'test_mae': round(test_mae, 4),
            'test_mse': round(test_mse, 4),
            'test_rmse': round(test_rmse, 4),
            'params': best_params
        }

        print(f"\n[{name}]")
        print(f"  CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f}) | CV RMSE: {cv_rmse_mean:.4f} | CV MAE: {cv_mae_mean:.4f}")
        print(f"  Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
        if best_params:
            print(f"  Tuned Parameters: {best_params}")

    print("\n=== Step 5: Model Selection ===")
    # Selection criteria: highest cross-validated R2 with regularization preference for generalization
    # Prefer regularized models (Ridge/Lasso/ElasticNet) if within margin
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]['cv_r2_mean'])
    print(f">> Best Model Selected: '{best_model_name}' with CV R² = {comparison_results[best_model_name]['cv_r2_mean']}")

    best_model = fitted_models[best_model_name]
    best_predictions = best_model.predict(X_test_processed)
    residuals = y_test - best_predictions

    # Extract coefficients and feature importance
    coefficients = best_model.coef_
    intercept = float(best_model.intercept_)
    feature_impacts = []
    for feat, coef in zip(engineered_feature_names, coefficients):
        direction = "Increases MPG" if coef > 0 else ("Decreases MPG" if coef < 0 else "Neutral")
        feature_impacts.append({
            'feature': feat,
            'coefficient': round(float(coef), 4),
            'abs_impact': round(float(abs(coef)), 4),
            'direction': direction
        })

    # Sort feature impacts by magnitude
    feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)

    print("\n=== Feature Coefficients & Relative Impact ===")
    for fi in feature_impacts:
        print(f"  - {fi['feature']:<25}: {fi['coefficient']:>8.4f} ({fi['direction']})")

    print("\n=== Step 6: Generating Visualizations ===")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # 1. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(y_test, best_predictions, color='#2563eb', alpha=0.75, edgecolors='none', s=45, label='Test Vehicles')
    min_val = min(y_test.min(), best_predictions.min()) - 2
    max_val = max(y_test.max(), best_predictions.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=1.8, label='Ideal Perfect Fit (y = x)')
    ax.set_title(f'Actual vs. Predicted MPG ({best_model_name})\nR² = {comparison_results[best_model_name]["test_r2"]:.3f}, RMSE = {comparison_results[best_model_name]["test_rmse"]:.2f}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Actual MPG', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted MPG', fontsize=11, fontweight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    actual_vs_pred_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
    fig.savefig(actual_vs_pred_path)
    plt.close(fig)

    # 2. Residual Plot
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(best_predictions, residuals, color='#0d9488', alpha=0.75, edgecolors='none', s=45)
    ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.8)
    ax.set_title(f'Residual Plot ({best_model_name})\nResiduals vs. Fitted Values', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted MPG', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    residual_path = os.path.join(PLOTS_DIR, 'residual_plot.png')
    fig.savefig(residual_path)
    plt.close(fig)

    # 3. Feature Impact Bar Chart
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    feat_names = [f['feature'] for f in reversed(feature_impacts)]
    feat_coefs = [f['coefficient'] for f in reversed(feature_impacts)]
    colors = ['#16a34a' if c > 0 else '#dc2626' for c in feat_coefs]
    ax.barh(feat_names, feat_coefs, color=colors, height=0.65)
    ax.axvline(0, color='#64748b', linestyle='-', linewidth=1)
    ax.set_title(f'Standardized Feature Coefficients ({best_model_name})', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Standardized Coefficient (Impact on MPG)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    coeff_path = os.path.join(PLOTS_DIR, 'feature_coefficients.png')
    fig.savefig(coeff_path)
    plt.close(fig)

    # 4. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    model_names = list(comparison_results.keys())
    r2_scores = [comparison_results[m]['test_r2'] for m in model_names]
    rmse_scores = [comparison_results[m]['test_rmse'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, r2_scores, width, label='Test R² Score', color='#3b82f6')
    ax.bar(x + width/2, rmse_scores, width, label='Test RMSE (MPG)', color='#f59e0b')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' Regression', '') for m in model_names], fontweight='bold')
    ax.set_title('Regression Model Comparison (Test Set)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True)
    plt.tight_layout()
    model_comp_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    fig.savefig(model_comp_path)
    plt.close(fig)

    print(f"Generated and saved evaluation plots to {PLOTS_DIR}")

    print("\n=== Step 7: Saving Model Artifacts & Metadata ===")
    # Save standalone preprocessing pipeline and regression model
    joblib.dump(preprocessing_pipeline, os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'regression_model.joblib'))
    
    # Save standalone scaler and individual estimators for versatile loader
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

    # Save model metadata
    metadata = {
        'project_name': 'Automobile Efficiency Regressor & Vehicle Class Profiler',
        'best_regression_model': best_model_name,
        'intercept': intercept,
        'model_comparison': comparison_results,
        'feature_names': list(engineered_feature_names),
        'feature_impacts': feature_impacts,
        'efficiency_thresholds': {
            'low_threshold': low_thresh,
            'high_threshold': high_thresh,
            'categories': {
                'low': f'<= {low_thresh:.1f} MPG (Bottom 33%)',
                'moderate': f'{low_thresh:.1f} - {high_thresh:.1f} MPG (Middle 34%)',
                'high': f'> {high_thresh:.1f} MPG (Top 33%)'
            }
        },
        'dataset_statistics': {
            'mpg': mpg_stats,
            'feature_medians': {col: float(df[col].median()) for col in ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']}
        },
        'reference_year': REFERENCE_YEAR,
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    print("Regression training completed successfully!")

if __name__ == '__main__':
    train_regression()
