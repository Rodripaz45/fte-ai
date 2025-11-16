# Pruebas Implementadas para el Microservicio FTE-AI

Se ha implementado una suite completa de pruebas siguiendo la metodología descrita en la documentación del proyecto, organizada en dos niveles complementarios:

## 1. Pruebas Técnicas del Servicio de IA

### Pruebas Unitarias (pytest)

#### `tests/unit/test_text_processing.py`
Valida las funciones encargadas de preparar y procesar el texto:

- **Limpieza y normalización**: 
  - Eliminación de caracteres innecesarios (teléfonos, correos, nombres, direcciones)
  - Unificación de mayúsculas y tildes
  - Producción de texto adecuado para vectorización
  - Preservación del contenido técnico relevante

- **Construcción del documento de entrada**:
  - Concatenación coherente del contenido del CV con los temas de talleres
  - Formato correcto para el modelo ML
  - Manejo de casos límite (CV vacío, sin talleres, etc.)

#### `tests/unit/test_model_inference.py`
Valida la función de inferencia del modelo:

- **Consistencia de resultados**:
  - La función devuelve siempre una lista de probabilidades
  - El tamaño coincide con el catálogo de competencias
  - Sin valores vacíos o inválidos
  - Probabilidades en rangos válidos [0, 1]

- **Umbral de decisión**:
  - Respeto al umbral configurado
  - Fallback cuando hay muy pocos resultados
  - Ordenamiento correcto por nivel descendente

- **Manejo de errores**:
  - Fallback a `decision_function` cuando no hay `predict_proba`
  - Manejo de texto vacío

### Pruebas de Integración

#### `tests/integration/test_analysis_service.py`
Pruebas sobre el servicio que orquesta el análisis de perfiles:

- **Contrato del payload**:
  - Validación de que el payload cumpla el contrato acordado
  - Verificación de identificador del participante, texto del CV, lista de talleres y banderas de configuración

- **Transformación de respuestas**:
  - Las respuestas del modelo se transforman correctamente en la estructura interna de perfil por competencias
  - Validación de campos: competencia, puntaje, nivel y confianza
  - Tipos y rangos correctos

- **Manejo de errores**:
  - Fallback a reglas cuando el ML no devuelve resultados
  - Manejo de casos con CV vacío o sin talleres
  - Metadata correcta según el modo utilizado (ML o reglas)

#### `tests/integration/test_api_endpoints.py`
Pruebas end-to-end de los endpoints utilizando TestClient de FastAPI:

- **Funcionamiento del endpoint**:
  - El endpoint `/analyze/profile` funciona correctamente
  - Validación de datos de entrada (Pydantic)
  - Respuestas con estructura correcta

- **Manejo de errores**:
  - Validación de campos requeridos
  - Validación de rangos (ej: asistencia_pct en [0, 1])
  - Manejo cuando el modelo no devuelve resultados

## 2. Pruebas de Calidad del Modelo de Clasificación

### `tests/model_quality/test_model_validation.py`

Evaluación empírica del modelo sobre un conjunto de validación manual:

- **Conjunto de validación manual**:
  - CVs representativos del contexto de la Fundación Trabajo Empresa
  - Perfiles esperados indicando qué competencias deberían estar presentes (según criterio humano)
  - 5 muestras iniciales cubriendo diferentes perfiles:
    - Ingeniero electromecánico (HSE, calidad, análisis de datos)
    - Desarrollador Full Stack (React, Node.js, DevOps)
    - Analista de datos (SQL, Python, ETL, BigQuery)
    - Especialista en seguridad (OWASP, SIEM)
    - Ingeniero de producción (Lean, ISO 9001)

- **Métricas calculadas**:
  - **Precisión**: Proporción de competencias predichas que resultan correctas
  - **Recall (Cobertura)**: Proporción de competencias esperadas que el modelo logra detectar
  - **F1-Score**: Media armónica de precisión y recall
  - **True Positives, False Positives, False Negatives**: Para análisis detallado

- **Reproducibilidad**:
  - Verificación de que los resultados sean reproducibles
  - Documentación del umbral de decisión utilizado

- **Umbral de decisión**:
  - Verificación de que el umbral esté documentado en los metadatos del modelo
  - Validación de que esté en el rango [0, 1]

## Estructura de Archivos

```
tests/
├── __init__.py
├── conftest.py                    # Fixtures compartidas
├── README.md                      # Documentación de las pruebas
├── unit/
│   ├── __init__.py
│   ├── test_text_processing.py    # Pruebas de procesamiento de texto
│   └── test_model_inference.py    # Pruebas de inferencia del modelo
├── integration/
│   ├── __init__.py
│   ├── test_analysis_service.py   # Pruebas del servicio
│   └── test_api_endpoints.py      # Pruebas de endpoints
└── model_quality/
    ├── __init__.py
    └── test_model_validation.py  # Pruebas de calidad del modelo
```

## Configuración

- **`pytest.ini`**: Configuración de pytest con marcadores y opciones
- **`requirements.txt`**: Actualizado con dependencias de testing:
  - `pytest==8.3.4`
  - `pytest-asyncio==0.24.0`
  - `pytest-cov==6.0.0`
  - `httpx==0.27.2` (requerido por TestClient)

## Ejecución

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
```

### Ejecutar con cobertura
```bash
pytest --cov=app --cov-report=html
```

### Usar el script auxiliar
```bash
python run_tests.py --type all --coverage --verbose
```

## Personalización del Conjunto de Validación

Para agregar nuevas muestras de validación manual, edita la constante `VALIDATION_DATASET` en `tests/model_quality/test_model_validation.py`. Cada entrada debe incluir:

- `id`: Identificador único
- `cv_texto`: Texto del CV de ejemplo
- `talleres`: Lista de talleres con tema y asistencia_pct
- `competencias_esperadas`: Set de competencias que deberían detectarse (según criterio humano)

## Notas Importantes

1. **Modelo requerido**: Las pruebas de calidad del modelo requieren que el modelo esté entrenado y disponible en `models/pipeline_competencias.joblib`

2. **Umbrales de calidad**: Los umbrales mínimos de precisión y recall (actualmente 30%) pueden ajustarse en `test_model_validation.py` según los requisitos del proyecto

3. **Reproducibilidad**: Las pruebas utilizan mocks y configuraciones determinísticas para asegurar reproducibilidad

4. **Mocks**: Las pruebas de integración utilizan mocks para evitar dependencias externas y permitir pruebas rápidas y aisladas

## Próximos Pasos Recomendados

1. **Ampliar el conjunto de validación**: Agregar más CVs representativos del contexto de la FTE
2. **Ajustar umbrales**: Refinar los umbrales de calidad según los resultados obtenidos
3. **Automatización**: Integrar las pruebas en un pipeline CI/CD
4. **Métricas adicionales**: Considerar agregar métricas como precisión@k o recall@k si es relevante

