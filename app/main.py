import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.database import init_db
from app.utils.model_loader import model_loader
from app.routes import health, prediction, profile, dashboard

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and verify model artifacts
    print("Starting Automobile Efficiency Regressor & Vehicle Class Profiler API...")
    init_db()
    model_loader.load_artifacts()
    yield
    print("Shutting down API service...")

app = FastAPI(
    title="Automobile Efficiency Regressor & Vehicle Class Profiler API",
    description="AI-driven automotive analytics API: Supervised regularized regression for MPG prediction and unsupervised KMeans clustering for vehicle archetype profiling.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(profile.router)
app.include_router(dashboard.router)

# Also mount under /api prefix for standard REST clients
app.include_router(health.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(profile.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")

# Custom validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", [])])
        errors.append({
            "field": field,
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "validation_error")
        })
    return JSONResponse(
        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
        content={
            "error": "Validation Error",
            "message": "One or more input fields failed validation.",
            "details": errors
        }
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred during request processing."
        }
    )

# Static Files & Frontend Serving
frontend_path = settings.FRONTEND_DIR
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(frontend_path / "index.html")

    @app.get("/prediction.html", include_in_schema=False)
    async def serve_prediction():
        return FileResponse(frontend_path / "prediction.html")

    @app.get("/profiles.html", include_in_schema=False)
    async def serve_profiles():
        return FileResponse(frontend_path / "profiles.html")

    @app.get("/analytics.html", include_in_schema=False)
    async def serve_analytics():
        return FileResponse(frontend_path / "analytics.html")

    @app.get("/history.html", include_in_schema=False)
    async def serve_history():
        return FileResponse(frontend_path / "history.html")

    @app.get("/about.html", include_in_schema=False)
    async def serve_about():
        return FileResponse(frontend_path / "about.html")
