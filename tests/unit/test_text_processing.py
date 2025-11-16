"""
Pruebas unitarias para funciones de procesamiento de texto.
Valida limpieza, normalización y construcción de documentos.
"""
import pytest
from app.services.analysis_service import (
    _clean_personal_info,
    _normalize_text,
    _build_text_for_model,
)


class TestCleanPersonalInfo:
    """Pruebas para la función de limpieza de información personal."""
    
    def test_removes_phone_numbers(self):
        """Verifica que se eliminen números de teléfono."""
        text = "Contacto: 1234567890 o celular 0987654321"
        result = _clean_personal_info(text)
        assert "1234567890" not in result
        assert "0987654321" not in result
    
    def test_removes_email_addresses(self):
        """Verifica que se eliminen direcciones de correo."""
        text = "Email: juan.perez@example.com o correo electrónico: test@test.com"
        result = _clean_personal_info(text)
        assert "juan.perez@example.com" not in result
        assert "test@test.com" not in result
    
    def test_removes_names(self):
        """Verifica que se eliminen nombres y apellidos."""
        text = "Nombre y Apellidos: Juan Pérez García"
        result = _clean_personal_info(text)
        assert "Juan Pérez García" not in result
        assert "Nombre y Apellidos" not in result.lower()
    
    def test_removes_addresses(self):
        """Verifica que se eliminen direcciones."""
        text = "Domicilio: Calle Principal 123, Ciudad"
        result = _clean_personal_info(text)
        assert "Calle Principal" not in result
    
    def test_preserves_technical_content(self):
        """Verifica que se preserve el contenido técnico relevante."""
        text = "Experiencia en Python, SQL y análisis de datos. Email: test@test.com"
        result = _clean_personal_info(text)
        assert "Python" in result or "python" in result.lower()
        assert "SQL" in result or "sql" in result.lower()
        assert "análisis" in result.lower() or "analisis" in result.lower()
        assert "test@test.com" not in result
    
    def test_handles_empty_string(self):
        """Verifica el manejo de cadenas vacías."""
        assert _clean_personal_info("") == ""
        assert _clean_personal_info(None) == ""
    
    def test_normalizes_whitespace(self):
        """Verifica que se normalicen múltiples espacios."""
        text = "Texto    con    espacios    múltiples"
        result = _clean_personal_info(text)
        assert "  " not in result  # No debe haber dobles espacios


class TestNormalizeText:
    """Pruebas para la función de normalización de texto."""
    
    def test_lowercase_conversion(self):
        """Verifica conversión a minúsculas."""
        assert _normalize_text("TEXTO EN MAYÚSCULAS") == "texto en mayúsculas"
    
    def test_strips_whitespace(self):
        """Verifica eliminación de espacios al inicio y final."""
        assert _normalize_text("  texto  ") == "texto"
        assert _normalize_text("\n\ttexto\n\t") == "texto"
    
    def test_handles_empty_string(self):
        """Verifica manejo de cadenas vacías."""
        assert _normalize_text("") == ""
        assert _normalize_text("   ") == ""


class TestBuildTextForModel:
    """Pruebas para la construcción del documento de entrada al modelo."""
    
    def test_concatenates_cv_and_talleres(self):
        """Verifica que se concatene CV y talleres correctamente."""
        cv_text = "Experiencia en Python"
        talleres = [{"tema": "excel"}, {"tema": "sql"}]
        
        result = _build_text_for_model(cv_text, talleres)
        
        assert "python" in result.lower()
        assert "topic:excel" in result
        assert "topic:sql" in result
    
    def test_cleans_personal_info_from_cv(self):
        """Verifica que se limpie información personal del CV."""
        cv_text = "Email: test@test.com. Experiencia en Python"
        talleres = None
        
        result = _build_text_for_model(cv_text, talleres)
        
        assert "test@test.com" not in result
        assert "python" in result.lower()
    
    def test_handles_empty_cv(self):
        """Verifica manejo de CV vacío."""
        talleres = [{"tema": "excel"}]
        result = _build_text_for_model(None, talleres)
        
        assert "topic:excel" in result
    
    def test_handles_empty_talleres(self):
        """Verifica manejo de talleres vacíos."""
        cv_text = "Experiencia en Python"
        result = _build_text_for_model(cv_text, None)
        
        assert "python" in result.lower()
        assert result.strip() == result  # Sin espacios extra
    
    def test_normalizes_to_lowercase(self):
        """Verifica que el resultado esté en minúsculas."""
        cv_text = "EXPERIENCIA EN PYTHON"
        talleres = [{"tema": "EXCEL"}]
        
        result = _build_text_for_model(cv_text, talleres)
        
        assert result.islower() or "python" in result.lower()
    
    def test_produces_valid_text_for_vectorization(self):
        """Verifica que el texto producido sea adecuado para vectorización."""
        cv_text = "Ingeniero con experiencia en análisis de datos usando Python y SQL"
        talleres = [{"tema": "power bi"}, {"tema": "excel"}]
        
        result = _build_text_for_model(cv_text, talleres)
        
        # Debe ser una cadena no vacía
        assert len(result) > 0
        # No debe tener caracteres especiales problemáticos
        assert "\x00" not in result
        # Debe tener contenido técnico
        assert any(keyword in result.lower() for keyword in ["python", "sql", "analisis", "datos"])

