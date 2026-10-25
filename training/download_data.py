import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'auto-mpg.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'auto-mpg-clean.csv')

def fetch_and_save_data():
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    
    print(f"Downloading Auto MPG dataset from {url}...")
    try:
        df = pd.read_csv(url, sep=r'\s+', names=cols, na_values='?')
    except Exception as e:
        print(f"Failed to fetch from URL ({e}), attempting backup source...")
        backup_url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
        df = pd.read_csv(backup_url)
        # Standardize column names if needed
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Save raw data
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset to {RAW_DATA_PATH} ({len(df)} records).")
    
    # Clean dataset
    df_clean = df.copy()
    
    # Convert horsepower to numeric (na_values handled)
    df_clean['horsepower'] = pd.to_numeric(df_clean['horsepower'], errors='coerce')
    
    # Impute missing horsepower using median of the dataset for processed reference copy
    median_hp = df_clean['horsepower'].median()
    df_clean['horsepower'] = df_clean['horsepower'].fillna(median_hp)
    
    # Save cleaned processed copy
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH} with imputed horsepower (median={median_hp}).")
    return df_clean

if __name__ == '__main__':
    fetch_and_save_data()
