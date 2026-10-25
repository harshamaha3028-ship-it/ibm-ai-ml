import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def create_nb1():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 01. Exploratory Data Analysis (Auto MPG Dataset)\n",
            "**Automobile Efficiency Regressor & Vehicle Class Profiler**\n",
            "This notebook explores distributions, correlations, and handles missing horsepower values."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Load Auto MPG dataset\n",
            "cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']\n",
            "df = pd.read_csv('../data/raw/auto-mpg.csv')\n",
            "print(f'Dataset Shape: {df.shape}')\n",
            "df.head()"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Data types and missing values summary\n",
            "print(df.info())\n",
            "print('\\nMissing values in horsepower: ', df['horsepower'].isna().sum() if df['horsepower'].dtype != 'O' else (df['horsepower'] == '?').sum())"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Convert horsepower to numeric and median impute\n",
            "df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')\n",
            "median_hp = df['horsepower'].median()\n",
            "df['horsepower'] = df['horsepower'].fillna(median_hp)\n",
            "print(f'Imputed horsepower with median: {median_hp} HP')\n",
            "df.describe()"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Distribution of Target Variable: MPG\n",
            "plt.figure(figsize=(8, 4))\n",
            "sns.histplot(df['mpg'], kde=True, color='#2563eb', bins=25)\n",
            "plt.title('Distribution of Target Variable: Miles Per Gallon (MPG)', fontweight='bold')\n",
            "plt.xlabel('MPG')\n",
            "plt.show()"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Correlation Matrix Heatmap\n",
            "plt.figure(figsize=(9, 7))\n",
            "numeric_df = df.drop(columns=['car_name'])\n",
            "sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
            "plt.title('Correlation Heatmap of Automobile Characteristics', fontweight='bold')\n",
            "plt.show()"
        ]}
    ]
    path = os.path.join(NOTEBOOKS_DIR, '01_data_exploration.ipynb')
    with open(path, 'w') as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created {path}")

def create_nb2():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 02. Feature Engineering\n",
            "**Automobile Efficiency Regressor & Vehicle Class Profiler**\n",
            "Deriving domain-specific automotive features without data leakage."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import sys\n",
            "import os\n",
            "\n",
            "sys.path.insert(0, os.path.abspath('..'))\n",
            "from app.utils.preprocessing import AutoMPGFeatureEngineer\n",
            "\n",
            "df = pd.read_csv('../data/raw/auto-mpg.csv')\n",
            "df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')\n",
            "df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())\n",
            "\n",
            "fe = AutoMPGFeatureEngineer(reference_year=82)\n",
            "df_feat = fe.transform(df)\n",
            "df_feat['mpg'] = df['mpg']\n",
            "df_feat.head()"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Correlation of engineered features with MPG\n",
            "corrs = df_feat.corr()['mpg'].sort_values()\n",
            "print('Correlation of features with MPG:\\n', corrs)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "corrs.drop('mpg').plot(kind='barh', color='#3b82f6')\n",
            "plt.title('Engineered Features Correlation with MPG', fontweight='bold')\n",
            "plt.xlabel('Pearson Correlation Coefficient')\n",
            "plt.show()"
        ]}
    ]
    path = os.path.join(NOTEBOOKS_DIR, '02_feature_engineering.ipynb')
    with open(path, 'w') as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created {path}")

def create_nb3():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 03. Supervised Regression Analysis & Regularization\n",
            "**Automobile Efficiency Regressor & Vehicle Class Profiler**\n",
            "Comparing Linear, Ridge, Lasso, and ElasticNet Regression with 5-Fold Cross Validation."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import json\n",
            "\n",
            "with open('../models/model_metadata.json', 'r') as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "print(f\"Selected Best Model: {meta['best_regression_model']}\")\n",
            "pd.DataFrame(meta['model_comparison']).T"
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# Feature coefficients\n",
            "impacts_df = pd.DataFrame(meta['feature_impacts'])\n",
            "print(impacts_df)\n",
            "\n",
            "plt.figure(figsize=(9, 5))\n",
            "colors = ['#16a34a' if c > 0 else '#dc2626' for c in impacts_df['coefficient']]\n",
            "plt.barh(impacts_df['feature'], impacts_df['coefficient'], color=colors)\n",
            "plt.axvline(0, color='grey', linestyle='--')\n",
            "plt.title('Standardized Coefficients (Feature Impact on MPG)', fontweight='bold')\n",
            "plt.xlabel('Coefficient Value')\n",
            "plt.gca().invert_yaxis()\n",
            "plt.show()"
        ]}
    ]
    path = os.path.join(NOTEBOOKS_DIR, '03_regression_analysis.ipynb')
    with open(path, 'w') as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created {path}")

def create_nb4():
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# 04. Unsupervised Vehicle Clustering & Profiling\n",
            "**Automobile Efficiency Regressor & Vehicle Class Profiler**\n",
            "KMeans clustering, Elbow method, Silhouette score, 2D PCA projection, and archetype discovery."
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import json\n",
            "\n",
            "with open('../models/model_metadata.json', 'r') as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "c_meta = meta['clustering']\n",
            "print(f\"Number of clusters: {c_meta['num_clusters']}\")\n",
            "print(f\"PCA Explained Variance: {c_meta['pca_explained_variance_ratio']}\")\n",
            "\n",
            "for cid, p in c_meta['cluster_profiles'].items():\n",
            "    print(f\"\\nCluster {cid}: {p['profile_name']} ({p['sample_count']} vehicles)\")\n",
            "    print(f\"  Avg MPG: {p['average_mpg']}, Weight: {p['average_weight']} lbs, HP: {p['average_horsepower']} hp, Cylinders: {p['average_cylinders']}\")"
        ]}
    ]
    path = os.path.join(NOTEBOOKS_DIR, '04_vehicle_clustering.ipynb')
    with open(path, 'w') as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created {path}")

if __name__ == '__main__':
    create_nb1()
    create_nb2()
    create_nb3()
    create_nb4()
