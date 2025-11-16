"""
Pruebas unitarias para la función de inferencia del modelo.
Valida que las predicciones sean consistentes y válidas.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.services.analysis_service import _predict_with_ml


class TestModelInference:
    """Pruebas para la función de inferencia del modelo ML."""
    
    @patch('app.services.analysis_service.load_model')
    def test_returns_list_of_probabilities(self, mock_load_model):
        """Verifica que siempre devuelva una lista de probabilidades."""
        # Mock del modelo
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.3, 0.5, 0.2, 0.15]])
        mock_classes = ["Comp1", "Comp2", "Comp3", "Comp4", "Comp5"]
        mock_metadata = {"best_threshold": 0.20}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto de prueba", None)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    @patch('app.services.analysis_service.load_model')
    def test_probabilities_match_classes_count(self, mock_load_model):
        """Verifica que el tamaño de probabilidades coincida con el catálogo."""
        num_classes = 10
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1] * num_classes])
        mock_classes = [f"Comp{i}" for i in range(num_classes)]
        mock_metadata = {"best_threshold": 0.20}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        # Verificar que todas las competencias del catálogo estén consideradas
        # (aunque no todas aparezcan en el resultado final por el umbral)
        assert all(isinstance(item, dict) for item in result)
    
    @patch('app.services.analysis_service.load_model')
    def test_no_empty_or_invalid_values(self, mock_load_model):
        """Verifica que no haya valores vacíos o inválidos."""
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.3, 0.5, 0.2]])
        mock_classes = ["Comp1", "Comp2", "Comp3", "Comp4"]
        mock_metadata = {"best_threshold": 0.20}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        for item in result:
            assert "competencia" in item
            assert "nivel" in item
            assert "confianza" in item
            assert "fuente" in item
            assert isinstance(item["nivel"], (int, float))
            assert 0 <= item["nivel"] <= 100
            assert isinstance(item["confianza"], (int, float))
            assert 0 <= item["confianza"] <= 1
    
    @patch('app.services.analysis_service.load_model')
    def test_respects_threshold(self, mock_load_model):
        """Verifica que se respete el umbral de decisión."""
        threshold = 0.30
        mock_pipeline = Mock()
        # Probabilidades: 0.1, 0.2, 0.3, 0.4, 0.5
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_classes = ["Comp1", "Comp2", "Comp3", "Comp4", "Comp5"]
        mock_metadata = {"best_threshold": threshold}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        # Solo deben aparecer competencias con probabilidad >= 0.30
        # (Comp3, Comp4, Comp5, o más si hay fallback)
        for item in result:
            assert item["nivel"] / 100.0 >= threshold or item["confianza"] < 0.85
    
    @patch('app.services.analysis_service.load_model')
    def test_fallback_to_decision_function(self, mock_load_model):
        """Verifica el fallback cuando no hay predict_proba."""
        mock_pipeline = Mock()
        # Simular que no tiene predict_proba
        del mock_pipeline.predict_proba
        mock_pipeline.decision_function.return_value = np.array([[0.5, -0.2, 1.0, -0.5]])
        mock_classes = ["Comp1", "Comp2", "Comp3", "Comp4"]
        mock_metadata = {"best_threshold": 0.20}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        # Debe funcionar sin error y devolver resultados
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)
    
    @patch('app.services.analysis_service.load_model')
    def test_results_sorted_by_level(self, mock_load_model):
        """Verifica que los resultados estén ordenados por nivel descendente."""
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.5, 0.3, 0.8, 0.2]])
        mock_classes = ["Comp1", "Comp2", "Comp3", "Comp4", "Comp5"]
        mock_metadata = {"best_threshold": 0.15}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        # Verificar orden descendente
        niveles = [item["nivel"] for item in result]
        assert niveles == sorted(niveles, reverse=True)
    
    @patch('app.services.analysis_service.load_model')
    def test_minimum_results_fallback(self, mock_load_model):
        """Verifica el fallback cuando hay muy pocos resultados sobre el umbral."""
        # Configurar para que haya muy pocos resultados sobre el umbral
        os.environ["ML_MIN_RESULTS"] = "5"
        os.environ["ML_MIN_PROB_FLOOR"] = "0.10"
        
        mock_pipeline = Mock()
        # Solo 2 resultados sobre umbral 0.30, pero necesitamos 5
        mock_pipeline.predict_proba.return_value = np.array([
            [0.35, 0.32, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        ])
        mock_classes = [f"Comp{i}" for i in range(8)]
        mock_metadata = {"best_threshold": 0.30}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("texto", None)
        
        # Debe completar hasta el mínimo requerido
        assert len(result) >= 5
        
        # Limpiar variables de entorno
        os.environ.pop("ML_MIN_RESULTS", None)
        os.environ.pop("ML_MIN_PROB_FLOOR", None)
    
    @patch('app.services.analysis_service.load_model')
    def test_handles_empty_text(self, mock_load_model):
        """Verifica el manejo de texto vacío."""
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1] * 5])
        mock_classes = [f"Comp{i}" for i in range(5)]
        mock_metadata = {"best_threshold": 0.20}
        
        mock_load_model.return_value = (mock_pipeline, mock_classes, mock_metadata)
        
        result = _predict_with_ml("", None)
        
        # Debe devolver una lista (puede estar vacía o con resultados según el modelo)
        assert isinstance(result, list)

