import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'auto-mpg.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'frontend', 'assets', 'plots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_profile_name(stats, dataset_medians):
    """
    Dynamically generates meaningful, human-readable profile titles based strictly
    on calculated cluster centroid statistics compared to dataset medians.
    """
    weight = stats['average_weight']
    hp = stats['average_horsepower']
    mpg = stats['average_mpg']
    cyl = stats['average_cylinders']
    
    med_weight = dataset_medians['weight']
    med_hp = dataset_medians['horsepower']
    med_mpg = dataset_medians['mpg']
    
    tags = []
    if weight < med_weight * 0.85 and mpg > med_mpg * 1.15:
        return "Lightweight Efficient"
    elif weight > med_weight * 1.20 and hp > med_hp * 1.25:
        return "Heavy High-Power"
    elif cyl >= 7 or hp > med_hp * 1.3:
        return "High-Performance V8"
    elif weight <= med_weight and mpg >= med_mpg:
        return "Balanced Economy"
    elif weight > med_weight and mpg < med_mpg:
        return "Mid-Heavy Cruiser"
    else:
        return "Balanced Midsize"

def train_clustering(optimal_k=3):
    print("=== Step 1: Loading and Preparing Data for Clustering ===")
    df = pd.read_csv(DATA_PATH)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

    clustering_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'mpg']
    X_cluster = df[clustering_features].copy()

    dataset_medians = {col: float(df[col].median()) for col in clustering_features}
    dataset_means = {col: float(df[col].mean()) for col in clustering_features}

    print(f"Clustering on {len(X_cluster)} samples using features: {clustering_features}")

    print("\n=== Step 2: Scaling Features ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    print("\n=== Step 3: Evaluating Optimal K (Elbow & Silhouette) ===")
    k_range = list(range(2, 9))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        sil_score = float(silhouette_score(X_scaled, km.labels_))
        silhouette_scores.append(sil_score)
        print(f"  k={k}: Inertia = {km.inertia_:.2f} | Silhouette Score = {sil_score:.4f}")

    # Determine optimal k based on highest silhouette or specified optimal_k
    best_k_idx = int(np.argmax(silhouette_scores))
    suggested_k = k_range[best_k_idx]
    chosen_k = optimal_k if optimal_k is not None else suggested_k
    print(f"\n>> Using k = {chosen_k} for vehicle profiling (Highest Silhouette at k={suggested_k})")

    # Fit final KMeans model
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['cluster'] = cluster_labels

    print("\n=== Step 4: 2D Dimensionality Reduction with PCA ===")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df['pca_x'] = pca_coords[:, 0]
    df['pca_y'] = pca_coords[:, 1]
    
    explained_var = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1 = {explained_var[0]*100:.1f}%, PC2 = {explained_var[1]*100:.1f}% (Total = {sum(explained_var)*100:.1f}%)")

    print("\n=== Step 5: Characterizing Clusters & Generating Dynamic Profiles ===")
    cluster_profiles = {}
    
    for c_id in range(chosen_k):
        c_df = df[df['cluster'] == c_id]
        
        avg_stats = {
            'cluster_id': int(c_id),
            'sample_count': int(len(c_df)),
            'percentage_of_fleet': round(float(len(c_df) / len(df) * 100), 1),
            'average_mpg': round(float(c_df['mpg'].mean()), 2),
            'average_weight': round(float(c_df['weight'].mean()), 1),
            'average_horsepower': round(float(c_df['horsepower'].mean()), 1),
            'average_displacement': round(float(c_df['displacement'].mean()), 1),
            'average_cylinders': round(float(c_df['cylinders'].mean()), 1),
            'average_acceleration': round(float(c_df['acceleration'].mean()), 2),
            'pca_centroid': [round(float(c_df['pca_x'].mean()), 3), round(float(c_df['pca_y'].mean()), 3)]
        }
        
        profile_name = generate_profile_name(avg_stats, dataset_medians)
        avg_stats['profile_name'] = profile_name
        
        # Descriptions
        characteristics = []
        if avg_stats['average_weight'] < dataset_means['weight']:
            characteristics.append(f"Low vehicle weight (~{avg_stats['average_weight']} lbs)")
        else:
            characteristics.append(f"Substantial vehicle weight (~{avg_stats['average_weight']} lbs)")

        if avg_stats['average_horsepower'] < dataset_means['horsepower']:
            characteristics.append(f"Modest horsepower (~{avg_stats['average_horsepower']} hp)")
        else:
            characteristics.append(f"High horsepower output (~{avg_stats['average_horsepower']} hp)")

        if avg_stats['average_mpg'] > dataset_means['mpg']:
            characteristics.append(f"High fuel efficiency (~{avg_stats['average_mpg']} MPG)")
        else:
            characteristics.append(f"Lower fuel economy (~{avg_stats['average_mpg']} MPG)")
            
        characteristics.append(f"Typical engine: {round(avg_stats['average_cylinders'])} cylinders, {avg_stats['average_displacement']} cu.in.")
        avg_stats['key_characteristics'] = characteristics

        cluster_profiles[str(c_id)] = avg_stats
        
        print(f"\n--- Cluster {c_id}: {profile_name} ({avg_stats['sample_count']} vehicles, {avg_stats['percentage_of_fleet']}%) ---")
        print(f"  Avg MPG: {avg_stats['average_mpg']} | Weight: {avg_stats['average_weight']} lbs | HP: {avg_stats['average_horsepower']} | Cylinders: {avg_stats['average_cylinders']}")
        for ch in characteristics:
            print(f"    * {ch}")

    print("\n=== Step 6: Generating Clustering Visualizations ===")
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Elbow Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8, color='#2563eb')
    ax.set_title('Elbow Method For Optimal k', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
    ax.set_xticks(k_range)
    plt.tight_layout()
    elbow_path = os.path.join(PLOTS_DIR, 'elbow_method.png')
    fig.savefig(elbow_path)
    plt.close(fig)

    # 2. Silhouette Score Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8, color='#16a34a')
    ax.set_title('Silhouette Analysis For Optimal k', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax.set_xticks(k_range)
    plt.tight_layout()
    sil_path = os.path.join(PLOTS_DIR, 'silhouette_scores.png')
    fig.savefig(sil_path)
    plt.close(fig)

    # 3. 2D PCA Cluster Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ['#2563eb', '#16a34a', '#f59e0b', '#dc2626', '#8b5cf6']
    for c_id in range(chosen_k):
        sub = df[df['cluster'] == c_id]
        p_name = cluster_profiles[str(c_id)]['profile_name']
        ax.scatter(sub['pca_x'], sub['pca_y'], color=colors[c_id % len(colors)], label=f"Cluster {c_id}: {p_name}", alpha=0.75, s=50, edgecolors='none')
        # Centroid marker
        cent = cluster_profiles[str(c_id)]['pca_centroid']
        ax.scatter(cent[0], cent[1], color='#0f172a', marker='X', s=160, edgecolors='white', linewidth=1.5)

    ax.set_title(f'Vehicle Clusters in 2D PCA Space (k={chosen_k})\nVariance Explained: {sum(explained_var)*100:.1f}%', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.legend(frameon=True, loc='best')
    plt.tight_layout()
    pca_plot_path = os.path.join(PLOTS_DIR, 'cluster_pca_scatter.png')
    fig.savefig(pca_plot_path)
    plt.close(fig)

    print(f"Generated and saved clustering plots to {PLOTS_DIR}")

    print("\n=== Step 7: Saving Artifacts and Updating Metadata ===")
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'clustering_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'cluster_scaler.joblib'))
    joblib.dump(pca, os.path.join(MODELS_DIR, 'pca.joblib'))

    # Update metadata JSON
    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata['clustering'] = {
        'num_clusters': chosen_k,
        'features_used': clustering_features,
        'elbow_inertias': {f'k={k}': round(v, 2) for k, v in zip(k_range, inertias)},
        'silhouette_scores': {f'k={k}': round(v, 4) for k, v in zip(k_range, silhouette_scores)},
        'pca_explained_variance_ratio': [round(float(v), 4) for v in explained_var],
        'cluster_profiles': cluster_profiles,
        'cluster_centroids_scaled': kmeans.cluster_centers_.tolist()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated metadata with clustering profiles in {metadata_path}")
    print("Clustering training completed successfully!")

if __name__ == '__main__':
    train_clustering()
