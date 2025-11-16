"""
Pruebas de integración para el servicio de análisis.
Valida el flujo completo desde el payload hasta la respuesta.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.analysis_service import analyze_participant_profile
from app.routes.analyze_routes import AnalyzeInput, TallerLite


class TestAnalysisServiceIntegration:
    """Pruebas de integración del servicio de análisis."""
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_payload_contract_compliance(self, mock_predict_ml):
        """Verifica que el payload cumpla el contrato acordado."""
        # Mock de respuesta del modelo
        mock_predict_ml.return_value = [
            {"competencia": "Analisis de Datos", "nivel": 85.0, "confianza": 0.85, "fuente": ["ml"]}
        ]
        
        # Crear payload con todos los campos requeridos
        talleres = [
            TallerLite(tema="excel", asistencia_pct=0.8),
            TallerLite(tema="python", asistencia_pct=0.9),
        ]
        payload = AnalyzeInput(
            participanteId="test-123",
            talleres=talleres,
            cvTexto="Ingeniero con experiencia en Python y SQL"
        )
        
        result = analyze_participant_profile(payload)
        
        # Verificar estructura de respuesta
        assert "participanteId" in result
        assert "competencias" in result
        assert "meta" in result
        assert result["participanteId"] == "test-123"
        
        # Verificar que se llamó al modelo con los datos correctos
        mock_predict_ml.assert_called_once()
        call_args = mock_predict_ml.call_args
        assert call_args[0][0] == payload.cvTexto  # CV texto
        assert call_args[0][1] is not None  # Talleres
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_response_structure_transformation(self, mock_predict_ml):
        """Verifica que las respuestas del modelo se transformen correctamente."""
        # Mock de respuesta del modelo
        mock_predict_ml.return_value = [
            {"competencia": "Analisis de Datos", "nivel": 85.5, "confianza": 0.85, "fuente": ["ml"]},
            {"competencia": "Ingeniería de Software", "nivel": 75.2, "confianza": 0.85, "fuente": ["ml"]},
        ]
        
        payload = AnalyzeInput(
            participanteId="test-456",
            cvTexto="Experiencia en análisis de datos y desarrollo de software"
        )
        
        result = analyze_participant_profile(payload)
        
        # Verificar estructura interna de perfil por competencias
        assert len(result["competencias"]) == 2
        
        for comp in result["competencias"]:
            assert "competencia" in comp
            assert "nivel" in comp
            assert "confianza" in comp
            assert "fuente" in comp
            
            # Verificar tipos y rangos
            assert isinstance(comp["competencia"], str)
            assert isinstance(comp["nivel"], (int, float))
            assert 0 <= comp["nivel"] <= 100
            assert isinstance(comp["confianza"], (int, float))
            assert 0 <= comp["confianza"] <= 1
            assert isinstance(comp["fuente"], list)
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_fallback_to_rules_when_ml_fails(self, mock_predict_ml):
        """Verifica el fallback a reglas cuando el modelo no devuelve resultados."""
        # Simular que el modelo no devuelve resultados
        mock_predict_ml.return_value = []
        
        talleres = [
            TallerLite(tema="excel", asistencia_pct=0.8),
        ]
        payload = AnalyzeInput(
            participanteId="test-789",
            talleres=talleres,
            cvTexto="Experiencia en Excel"
        )
        
        result = analyze_participant_profile(payload)
        
        # Debe usar modo de reglas
        assert result["meta"]["mode"] == "rules"
        assert "competencias" in result
        assert len(result["competencias"]) > 0
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_handles_missing_cv_text(self, mock_predict_ml):
        """Verifica el manejo cuando falta el texto del CV."""
        mock_predict_ml.return_value = []
        
        talleres = [TallerLite(tema="python", asistencia_pct=0.9)]
        payload = AnalyzeInput(
            participanteId="test-no-cv",
            talleres=talleres,
            cvTexto=None
        )
        
        result = analyze_participant_profile(payload)
        
        # Debe procesar solo con talleres
        assert result["participanteId"] == "test-no-cv"
        assert "competencias" in result
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_handles_missing_talleres(self, mock_predict_ml):
        """Verifica el manejo cuando faltan talleres."""
        mock_predict_ml.return_value = []
        
        payload = AnalyzeInput(
            participanteId="test-no-talleres",
            talleres=None,
            cvTexto="Experiencia en Python y SQL"
        )
        
        result = analyze_participant_profile(payload)
        
        # Debe procesar solo con CV
        assert result["participanteId"] == "test-no-talleres"
        assert "competencias" in result
    
    @patch('app.services.analysis_service._predict_with_ml')
    def test_ml_mode_metadata(self, mock_predict_ml):
        """Verifica que se incluya metadata cuando se usa ML."""
        mock_predict_ml.return_value = [
            {"competencia": "Analisis de Datos", "nivel": 80.0, "confianza": 0.85, "fuente": ["ml"]}
        ]
        
        payload = AnalyzeInput(
            participanteId="test-ml",
            cvTexto="Análisis de datos"
        )
        
        result = analyze_participant_profile(payload)
        
        assert result["meta"]["mode"] == "ml"

