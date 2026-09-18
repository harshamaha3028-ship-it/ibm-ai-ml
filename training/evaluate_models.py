import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.json')

def print_evaluation_summary():
    if not os.path.exists(METADATA_PATH):
        print("Model metadata not found. Please train models first using `python training/train_regression.py`.")
        return

    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)

    print("\n" + "="*70)
    print("      AUTOMOBILE EFFICIENCY REGRESSOR & VEHICLE PROFILER")
    print("                    EVALUATION SUMMARY")
    print("="*70)

    print(f"\n[1] SUPERVISED REGRESSION BENCHMARK")
    print(f"Selected Best Model: {meta.get('best_regression_model')}")
    print("-" * 70)
    print(f"{'Model':<22} | {'Train R²':<9} | {'CV R²':<9} | {'Test R²':<9} | {'Test RMSE':<9} | {'Test MAE':<9}")
    print("-" * 70)

    comp = meta.get('model_comparison', {})
    for m_name, metrics in comp.items():
        print(f"{m_name:<22} | {metrics['train_r2']:<9.4f} | {metrics['cv_r2_mean']:<9.4f} | {metrics['test_r2']:<9.4f} | {metrics['test_rmse']:<9.4f} | {metrics['test_mae']:<9.4f}")
    print("-" * 70)

    print(f"\n[2] TOP PREDICTIVE FEATURE COEFFICIENTS")
    impacts = meta.get('feature_impacts', [])
    for fi in impacts[:8]:
        direction_icon = "(+) Increases MPG" if fi['coefficient'] > 0 else "(-) Decreases MPG"
        print(f"  * {fi['feature']:<25}: {fi['coefficient']:>8.4f}  {direction_icon}")

    print(f"\n[3] DATASET-DERIVED MPG EFFICIENCY CATEGORIES")
    eff = meta.get('efficiency_thresholds', {}).get('categories', {})
    for cat, desc in eff.items():
        print(f"  * {cat.capitalize():<10}: {desc}")

    if 'clustering' in meta:
        print(f"\n[4] VEHICLE CLUSTERING ARCHETYPES (k={meta['clustering']['num_clusters']})")
        profiles = meta['clustering'].get('cluster_profiles', {})
        for cid, prof in profiles.items():
            print(f"  Cluster {cid}: {prof['profile_name']}")
            print(f"    - Vehicles: {prof['sample_count']} ({prof['percentage_of_fleet']}%)")
            print(f"    - Avg MPG: {prof['average_mpg']} | Weight: {prof['average_weight']} lbs | HP: {prof['average_horsepower']} hp | Cylinders: {prof['average_cylinders']}")
            for ch in prof['key_characteristics']:
                print(f"      * {ch}")
    print("\n" + "="*70)

if __name__ == '__main__':
    print_evaluation_summary()
