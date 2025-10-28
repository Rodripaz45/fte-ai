from fastapi import FastAPI
from app.routes.health_routes import router as health_router
from app.routes.analyze_routes import router as analyze_router
from app.routes.job_routes import router as job_router

app = FastAPI(
    title="FTE-AI",
    description="Microservicio de análisis de competencias y perfilado de participantes para la FTE.",
    version="1.0.0",
)

# Registrar rutas
app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(job_router)
