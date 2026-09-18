import os
import json
import sqlite3
import datetime
from typing import List, Dict, Any, Optional
from app.config import settings

# Global connection flag
_IS_MYSQL = False

def get_mysql_connection():
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            charset='utf8mb4'
        )
        return conn
    except Exception as e:
        return None

def get_sqlite_connection():
    conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_db_connection():
    """
    Returns a live database connection (MySQL if available, otherwise SQLite fallback).
    """
    global _IS_MYSQL
    if settings.DB_TYPE.lower() == "mysql":
        conn = get_mysql_connection()
        if conn:
            _IS_MYSQL = True
            return conn
        raise ConnectionError("Failed to connect to MySQL database as configured.")
    elif settings.DB_TYPE.lower() == "sqlite":
        _IS_MYSQL = False
        return get_sqlite_connection()
    else:
        # Auto mode: try MySQL first, fallback to SQLite
        mysql_conn = get_mysql_connection()
        if mysql_conn is not None:
            _IS_MYSQL = True
            return mysql_conn
        _IS_MYSQL = False
        return get_sqlite_connection()

def is_using_mysql() -> bool:
    return _IS_MYSQL

def init_db():
    """
    Initializes database tables and seeds initial metrics and cluster profiles from metadata.
    """
    global _IS_MYSQL
    conn = get_db_connection()
    cursor = conn.cursor()

    if _IS_MYSQL:
        print("[Database] Initializing MySQL tables...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cylinders INT NOT NULL,
            displacement FLOAT NOT NULL,
            horsepower FLOAT NOT NULL,
            weight FLOAT NOT NULL,
            acceleration FLOAT NOT NULL,
            model_year INT NOT NULL,
            origin VARCHAR(32) NOT NULL,
            predicted_mpg FLOAT NOT NULL,
            efficiency_category VARCHAR(64) NOT NULL,
            cluster_id INT NOT NULL,
            profile_name VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(128) NOT NULL,
            mae FLOAT NOT NULL,
            mse FLOAT NOT NULL,
            rmse FLOAT NOT NULL,
            r2 FLOAT NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_profiles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cluster_id INT NOT NULL UNIQUE,
            profile_name VARCHAR(128) NOT NULL,
            average_mpg FLOAT NOT NULL,
            average_weight FLOAT NOT NULL,
            average_horsepower FLOAT NOT NULL,
            average_displacement FLOAT NOT NULL,
            average_cylinders FLOAT NOT NULL,
            sample_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    else:
        print(f"[Database] Initializing SQLite fallback ({settings.SQLITE_PATH})...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cylinders INTEGER NOT NULL,
            displacement REAL NOT NULL,
            horsepower REAL NOT NULL,
            weight REAL NOT NULL,
            acceleration REAL NOT NULL,
            model_year INTEGER NOT NULL,
            origin TEXT NOT NULL,
            predicted_mpg REAL NOT NULL,
            efficiency_category TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            profile_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            mae REAL NOT NULL,
            mse REAL NOT NULL,
            rmse REAL NOT NULL,
            r2 REAL NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL UNIQUE,
            profile_name TEXT NOT NULL,
            average_mpg REAL NOT NULL,
            average_weight REAL NOT NULL,
            average_horsepower REAL NOT NULL,
            average_displacement REAL NOT NULL,
            average_cylinders REAL NOT NULL,
            sample_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

    conn.commit()

    # Seed model metrics and cluster profiles from model_metadata.json if available
    metadata_path = settings.MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)

            # Check if model_metrics is empty
            cursor.execute("SELECT COUNT(*) FROM model_metrics")
            res = cursor.fetchone()
            count = res[0] if res else 0
            if count == 0 and 'model_comparison' in meta:
                for m_name, vals in meta['model_comparison'].items():
                    sql = "INSERT INTO model_metrics (model_name, mae, mse, rmse, r2) VALUES (%s, %s, %s, %s, %s)" if _IS_MYSQL else "INSERT INTO model_metrics (model_name, mae, mse, rmse, r2) VALUES (?, ?, ?, ?, ?)"
                    cursor.execute(sql, (m_name, vals['test_mae'], vals['test_mse'], vals['test_rmse'], vals['test_r2']))

            # Check if cluster_profiles is empty
            cursor.execute("SELECT COUNT(*) FROM cluster_profiles")
            res = cursor.fetchone()
            count_c = res[0] if res else 0
            if count_c == 0 and 'clustering' in meta:
                c_profs = meta['clustering'].get('cluster_profiles', {})
                for c_id, prof in c_profs.items():
                    sql = """
                    INSERT INTO cluster_profiles (cluster_id, profile_name, average_mpg, average_weight, average_horsepower, average_displacement, average_cylinders, sample_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """ if _IS_MYSQL else """
                    INSERT INTO cluster_profiles (cluster_id, profile_name, average_mpg, average_weight, average_horsepower, average_displacement, average_cylinders, sample_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(sql, (
                        int(prof['cluster_id']),
                        prof['profile_name'],
                        float(prof['average_mpg']),
                        float(prof['average_weight']),
                        float(prof['average_horsepower']),
                        float(prof['average_displacement']),
                        float(prof['average_cylinders']),
                        int(prof.get('sample_count', 0))
                    ))
            conn.commit()
        except Exception as ex:
            print(f"[Database Warning] Auto-seeding metadata encountered: {ex}")

    cursor.close()
    conn.close()
    print("[Database] Schema ready.")

def save_prediction(data: Dict[str, Any]) -> int:
    """
    Inserts a vehicle prediction record using parameterized query.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    is_my = is_using_mysql()
    sql = """
    INSERT INTO predictions (
        cylinders, displacement, horsepower, weight, acceleration,
        model_year, origin, predicted_mpg, efficiency_category,
        cluster_id, profile_name, created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """ if is_my else """
    INSERT INTO predictions (
        cylinders, displacement, horsepower, weight, acceleration,
        model_year, origin, predicted_mpg, efficiency_category,
        cluster_id, profile_name, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = (
        int(data['cylinders']),
        float(data['displacement']),
        float(data['horsepower']),
        float(data['weight']),
        float(data['acceleration']),
        int(data['model_year']),
        str(data['origin']),
        float(data['predicted_mpg']),
        str(data['efficiency_category']),
        int(data['cluster_id']),
        str(data['profile_name']),
        now_str
    )
    
    cursor.execute(sql, params)
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return new_id

def get_predictions_history(
    search: Optional[str] = None,
    efficiency: Optional[str] = None,
    profile: Optional[str] = None,
    sort_by: str = "id",
    order: str = "desc",
    page: int = 1,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Retrieves filtered, sorted, paginated predictions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    is_my = is_using_mysql()
    
    # Whitelist allowed sort columns to prevent SQL injection
    allowed_sort_cols = {
        'id': 'id',
        'predicted_mpg': 'predicted_mpg',
        'efficiency_category': 'efficiency_category',
        'profile_name': 'profile_name',
        'created_at': 'created_at',
        'weight': 'weight',
        'horsepower': 'horsepower'
    }
    col = allowed_sort_cols.get(sort_by, 'id')
    direction = 'ASC' if order.lower() == 'asc' else 'DESC'

    conditions = []
    params = []

    if search:
        if is_my:
            conditions.append("(profile_name LIKE %s OR origin LIKE %s OR efficiency_category LIKE %s)")
            wild = f"%{search}%"
            params.extend([wild, wild, wild])
        else:
            conditions.append("(profile_name LIKE ? OR origin LIKE ? OR efficiency_category LIKE ?)")
            wild = f"%{search}%"
            params.extend([wild, wild, wild])

    if efficiency:
        if is_my:
            conditions.append("efficiency_category = %s")
            params.append(efficiency)
        else:
            conditions.append("efficiency_category = ?")
            params.append(efficiency)

    if profile:
        if is_my:
            conditions.append("profile_name = %s")
            params.append(profile)
        else:
            conditions.append("profile_name = ?")
            params.append(profile)

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    # Count total
    count_sql = f"SELECT COUNT(*) FROM predictions{where_clause}"
    cursor.execute(count_sql, tuple(params))
    res = cursor.fetchone()
    total_records = res[0] if res else 0

    # Paginate
    offset = (page - 1) * limit
    if is_my:
        data_sql = f"SELECT * FROM predictions{where_clause} ORDER BY {col} {direction} LIMIT %s OFFSET %s"
        exec_params = tuple(params + [limit, offset])
    else:
        data_sql = f"SELECT * FROM predictions{where_clause} ORDER BY {col} {direction} LIMIT ? OFFSET ?"
        exec_params = tuple(params + [limit, offset])

    cursor.execute(data_sql, exec_params)
    rows = cursor.fetchall()

    items = []
    if is_my:
        col_names = [d[0] for d in cursor.description]
        for row in rows:
            r_dict = dict(zip(col_names, row))
            if isinstance(r_dict.get('created_at'), datetime.datetime):
                r_dict['created_at'] = r_dict['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            items.append(r_dict)
    else:
        for row in rows:
            r_dict = dict(row)
            items.append(r_dict)

    cursor.close()
    conn.close()

    total_pages = max(1, (total_records + limit - 1) // limit)
    return {
        'items': items,
        'total': total_records,
        'page': page,
        'limit': limit,
        'total_pages': total_pages
    }

def delete_all_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    cursor.close()
    conn.close()
