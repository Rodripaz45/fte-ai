"""
Configuración compartida para todas las pruebas.
Incluye fixtures comunes y configuración de pytest.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_cv_text():
    """CV de ejemplo para pruebas."""
    return """
    Ingeniero electromecánico con experiencia en HSE, gestión de calidad, 
    supervisión de calibración, planificación logística y análisis de datos.
    Habilidades en Power BI, VBA y Microsoft Office 365.
    """

@pytest.fixture
def sample_cv_with_personal_info():
    """CV con información personal que debe ser limpiada."""
    return """
    Nombre y Apellidos: Juan Pérez García
    Correo electrónico: juan.perez@email.com
    Teléfono: 1234567890
    Celular: 0987654321
    Domicilio: Calle Principal 123, Ciudad
    
    Ingeniero con experiencia en Python y SQL.
    """

@pytest.fixture
def sample_talleres():
    """Lista de talleres de ejemplo para pruebas."""
    return [
        {"tema": "excel", "asistencia_pct": 0.8},
        {"tema": "python", "asistencia_pct": 0.9},
        {"tema": "sql", "asistencia_pct": 0.7},
    ]

@pytest.fixture
def sample_payload():
    """Payload de ejemplo para el endpoint de análisis."""
    from app.routes.analyze_routes import AnalyzeInput, TallerLite
    
    talleres = [
        TallerLite(tema="excel", asistencia_pct=0.8),
        TallerLite(tema="python", asistencia_pct=0.9),
    ]
    
    return AnalyzeInput(
        participanteId="test-123",
        talleres=talleres,
        cvTexto="Ingeniero con experiencia en Python, SQL y análisis de datos."
    )

@pytest.fixture
def mock_model_predictions():
    """Mock de predicciones del modelo ML."""
    return {
        "Analisis de Datos": 0.85,
        "Ingeniería de Software": 0.75,
        "Ofimática": 0.65,
        "DevOps/SRE": 0.30,
        "Ciberseguridad": 0.15,
    }

@pytest.fixture(autouse=True)
def reset_model_cache():
    """Resetea el cache del modelo antes de cada prueba."""
    from app.ml.model_loader import load_model
    load_model.cache_clear()
    yield
    load_model.cache_clear()

