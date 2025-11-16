"""
Pruebas de integración para los endpoints de la API.
Utiliza TestClient de FastAPI para pruebas end-to-end.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    """Cliente de prueba para la API."""
    return TestClient(app)


class TestAnalyzeEndpoints:
    """Pruebas para los endpoints de análisis."""
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_analyze_profile_endpoint_success(self, mock_predict_ml, client):
        """Verifica que el endpoint /analyze/profile funcione correctamente."""
        # Mock de respuesta del modelo
        mock_predict_ml.return_value = [
            {
                "competencia": "Analisis de Datos",
                "nivel": 85.0,
                "confianza": 0.85,
                "fuente": ["ml"]
            }
        ]
        
        payload = {
            "participanteId": "test-api-123",
            "talleres": [
                {"tema": "excel", "asistencia_pct": 0.8},
                {"tema": "python", "asistencia_pct": 0.9}
            ],
            "cvTexto": "Ingeniero con experiencia en Python y SQL"
        }
        
        response = client.post("/analyze/profile", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "participanteId" in data
        assert "competencias" in data
        assert data["participanteId"] == "test-api-123"
    
    def test_analyze_profile_endpoint_validation(self, client):
        """Verifica validación de datos en el endpoint."""
        # Payload inválido: asistencia_pct fuera de rango
        payload = {
            "participanteId": "test-invalid",
            "talleres": [
                {"tema": "excel", "asistencia_pct": 1.5}  # Fuera de rango [0, 1]
            ],
            "cvTexto": "Texto de prueba"
        }
        
        response = client.post("/analyze/profile", json=payload)
        
        # FastAPI debe rechazar el payload inválido
        assert response.status_code == 422
    
    def test_analyze_profile_endpoint_missing_fields(self, client):
        """Verifica manejo de campos faltantes."""
        # Payload sin participanteId requerido
        payload = {
            "talleres": [{"tema": "excel", "asistencia_pct": 0.8}]
        }
        
        response = client.post("/analyze/profile", json=payload)
        
        # Debe fallar la validación
        assert response.status_code == 422
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_analyze_profile_endpoint_empty_response(self, mock_predict_ml, client):
        """Verifica manejo cuando el modelo no devuelve resultados."""
        mock_predict_ml.return_value = []
        
        payload = {
            "participanteId": "test-empty",
            "cvTexto": "Texto sin competencias relevantes"
        }
        
        response = client.post("/analyze/profile", json=payload)
        
        # Debe responder con éxito pero usando fallback
        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["mode"] == "rules"

