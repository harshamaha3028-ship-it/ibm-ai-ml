from typing import Dict, Any, List, Optional
from app.schemas.vehicle import VehicleInput
from app.schemas.prediction import AnalysisResponse, FeatureContribution
from app.services.prediction_service import prediction_service
from app.services.profiling_service import profiling_service
from app.utils.model_loader import model_loader
from app.utils.preprocessing import ORIGIN_MAP
from app.database import save_prediction, get_predictions_history, get_db_connection, is_using_mysql

class AnalyticsService:
    def __init__(self):
        self.loader = model_loader

    def analyze_vehicle(self, vehicle: VehicleInput) -> AnalysisResponse:
        # Step 1: Predict MPG & explainability
        predicted_mpg, efficiency_category, contributions = prediction_service.predict_mpg(vehicle)

        # Step 2: Unsupervised Clustering & Profiling
        profile_res = profiling_service.profile_vehicle(vehicle, predicted_mpg)

        # Step 3: Comparative Deltas against Cluster Centroid
        char = profile_res.characteristics
        comparison = {
            'weight_diff': round(vehicle.weight - char.average_weight, 1),
            'horsepower_diff': round(vehicle.horsepower - char.average_horsepower, 1),
            'displacement_diff': round(vehicle.displacement - char.average_displacement, 1),
            'mpg_diff': round(predicted_mpg - char.average_mpg, 1),
            'is_lighter_than_cluster_avg': bool(vehicle.weight < char.average_weight),
            'is_more_efficient_than_cluster_avg': bool(predicted_mpg > char.average_mpg)
        }

        # Step 4: Persist in Database
        origin_label = ORIGIN_MAP.get(int(vehicle.origin), str(vehicle.origin))
        pred_id = None
        try:
            pred_id = save_prediction({
                'cylinders': vehicle.cylinders,
                'displacement': vehicle.displacement,
                'horsepower': vehicle.horsepower,
                'weight': vehicle.weight,
                'acceleration': vehicle.acceleration,
                'model_year': vehicle.model_year,
                'origin': origin_label,
                'predicted_mpg': predicted_mpg,
                'efficiency_category': efficiency_category,
                'cluster_id': profile_res.cluster_id,
                'profile_name': profile_res.profile_name
            })
        except Exception as e:
            print(f"[AnalyticsService Database Save Error] {e}")

        return AnalysisResponse(
            prediction_id=pred_id,
            predicted_mpg=predicted_mpg,
            efficiency_category=efficiency_category,
            vehicle_profile=profile_res.profile_name,
            cluster_id=profile_res.cluster_id,
            key_characteristics=profile_res.key_characteristics,
            feature_contributions=contributions,
            cluster_comparison=comparison,
            pca_coordinates=profile_res.pca_coordinates
        )

    def get_dashboard_statistics(self) -> Dict[str, Any]:
        meta = self.loader.metadata
        
        # Dataset stats
        ds_stats = meta.get('dataset_statistics', {}).get('mpg', {})
        best_model_name = meta.get('best_regression_model', 'Linear Regression')
        best_model_metrics = meta.get('model_comparison', {}).get(best_model_name, {})
        
        # Query database predictions count & breakdown
        total_predictions = 0
        efficiency_counts = {'Low Efficiency': 0, 'Moderate Efficiency': 0, 'High Efficiency': 0}
        recent_predictions = []
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            is_my = is_using_mysql()
            
            cursor.execute("SELECT COUNT(*) FROM predictions")
            res = cursor.fetchone()
            total_predictions = res[0] if res else 0

            cursor.execute("SELECT efficiency_category, COUNT(*) FROM predictions GROUP BY efficiency_category")
            rows = cursor.fetchall()
            for r in rows:
                cat_name = r[0]
                c_cnt = r[1]
                if cat_name in efficiency_counts:
                    efficiency_counts[cat_name] = c_cnt

            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
            rec_rows = cursor.fetchall()
            if is_my:
                col_names = [d[0] for d in cursor.description]
                for row in rec_rows:
                    recent_predictions.append(dict(zip(col_names, row)))
            else:
                for row in rec_rows:
                    recent_predictions.append(dict(row))

            cursor.close()
            conn.close()
        except Exception as ex:
            print(f"[Dashboard Query Warning] {ex}")

        clusters_meta = meta.get('clustering', {}).get('cluster_profiles', {})

        return {
            'dataset_summary': {
                'total_vehicles': ds_stats.get('count', 398),
                'average_mpg': round(ds_stats.get('mean', 23.51), 2),
                'min_mpg': ds_stats.get('min', 9.0),
                'max_mpg': ds_stats.get('max', 46.6),
                'median_mpg': ds_stats.get('median', 23.0)
            },
            'model_summary': {
                'best_model': best_model_name,
                'r2_score': best_model_metrics.get('test_r2', 0.875),
                'rmse': best_model_metrics.get('test_rmse', 2.58),
                'mae': best_model_metrics.get('test_mae', 1.94),
                'total_models_evaluated': len(meta.get('model_comparison', {}))
            },
            'clustering_summary': {
                'num_profiles': len(clusters_meta),
                'profiles': [v['profile_name'] for v in clusters_meta.values()]
            },
            'application_stats': {
                'total_predictions_made': total_predictions,
                'efficiency_distribution': efficiency_counts,
                'recent_predictions': recent_predictions
            }
        }

analytics_service = AnalyticsService()
