# Pruebas del Microservicio FTE-AI

Este directorio contiene las pruebas del microservicio de análisis de competencias, organizadas en tres niveles complementarios según la metodología descrita en la documentación del proyecto.

## Estructura de Pruebas

### 1. Pruebas Unitarias (`tests/unit/`)

Pruebas técnicas sobre funciones individuales de procesamiento de texto:

- **`test_text_processing.py`**: Valida funciones de limpieza, normalización y construcción de documentos
  - Limpieza de información personal (teléfonos, correos, nombres, direcciones)
  - Normalización de texto (mayúsculas, espacios)
  - Construcción del documento de entrada para el modelo

- **`test_model_inference.py`**: Valida la función de inferencia del modelo
  - Consistencia en el tamaño de probabilidades
  - Ausencia de valores vacíos o inválidos
  - Respeto al umbral de decisión
  - Fallback cuando no hay `predict_proba`

### 2. Pruebas de Integración (`tests/integration/`)

Pruebas sobre el servicio completo y los endpoints de la API:

- **`test_analysis_service.py`**: Valida el servicio de análisis
  - Cumplimiento del contrato del payload
  - Transformación correcta de respuestas del modelo
  - Fallback a reglas cuando el ML no devuelve resultados
  - Manejo de casos límite (CV vacío, sin talleres, etc.)

- **`test_api_endpoints.py`**: Pruebas end-to-end de los endpoints
  - Funcionamiento correcto del endpoint `/analyze/profile`
  - Validación de datos de entrada
  - Manejo de errores y casos excepcionales

### 3. Pruebas de Calidad del Modelo (`tests/model_quality/`)

Evaluación empírica del modelo sobre un conjunto de validación manual:

- **`test_model_validation.py`**: Evalúa precisión y recall
  - Conjunto de CVs representativos del contexto de la FTE
  - Comparación de competencias predichas vs esperadas
  - Cálculo de métricas (precisión, recall, F1)
  - Verificación de reproducibilidad
  - Documentación del umbral de decisión

## Ejecución de Pruebas

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar todas las pruebas

```bash
pytest
```

### Ejecutar por categoría

```bash
# Solo pruebas unitarias
pytest tests/unit/ -m unit

# Solo pruebas de integración
pytest tests/integration/ -m integration

# Solo pruebas de calidad del modelo
pytest tests/model_quality/ -m model_quality

# Excluir pruebas lentas
pytest -m "not slow"
```

### Ejecutar con cobertura

```bash
pytest --cov=app --cov-report=html
```

### Ejecutar con verbosidad

```bash
pytest -v
```

### Ejecutar una prueba específica

```bash
pytest tests/unit/test_text_processing.py::TestCleanPersonalInfo::test_removes_phone_numbers
```

## Configuración

La configuración de pytest se encuentra en `pytest.ini` en la raíz del proyecto.

### Marcadores disponibles

- `@pytest.mark.unit`: Pruebas unitarias
- `@pytest.mark.integration`: Pruebas de integración
- `@pytest.mark.model_quality`: Pruebas de calidad del modelo
- `@pytest.mark.slow`: Pruebas que tardan más tiempo

## Conjunto de Validación Manual

El conjunto de validación manual se define en `tests/model_quality/test_model_validation.py` en la constante `VALIDATION_DATASET`. Cada entrada incluye:

- `id`: Identificador único
- `cv_texto`: Texto del CV de ejemplo
- `talleres`: Lista de talleres asociados
- `competencias_esperadas`: Set de competencias que deberían detectarse (según criterio humano)

Para agregar nuevas muestras de validación, edita esta constante.

## Métricas de Calidad

Las pruebas de calidad calculan:

- **Precisión**: Proporción de competencias predichas que resultan correctas
- **Recall (Cobertura)**: Proporción de competencias esperadas que el modelo logra detectar
- **F1-Score**: Media armónica de precisión y recall

Estas métricas permiten cuantificar la utilidad del modelo en el contexto real de la FTE y sirven como referencia para futuros ciclos de mejora del entrenamiento.

## Reproducibilidad

Las pruebas fijan una semilla aleatoria (cuando aplica) para asegurar reproducibilidad. El umbral de decisión utilizado se documenta en los metadatos del modelo y se verifica en las pruebas.

## Notas

- Las pruebas de calidad del modelo requieren que el modelo esté entrenado y disponible en `models/pipeline_competencias.joblib`
- Algunas pruebas utilizan mocks para evitar dependencias externas
- Las pruebas de integración pueden requerir que el servicio esté configurado correctamente

