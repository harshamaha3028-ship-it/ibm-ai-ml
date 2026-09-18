import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
env_file = BASE_DIR / '.env'
if env_file.exists():
    load_dotenv(dotenv_path=env_file)

class Settings:
    PROJECT_NAME: str = "Automobile Efficiency Regressor & Vehicle Class Profiler"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    
    # Database Settings
    DB_TYPE: str = os.getenv("DB_TYPE", "auto") # auto, mysql, sqlite
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 3306))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "automobile_ml")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", str(BASE_DIR / "automobile_ml.sqlite3"))
    
    # Server Settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

settings = Settings()
