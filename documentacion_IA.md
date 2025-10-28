# Documentación Técnica - Proyecto de Inteligencia Artificial
## FTE-AI: Sistema de Análisis de Competencias con Machine Learning

---

## 1. DESCRIPCIÓN DEL DATASET

### 1.1 Origen y Propósito
- **Dataset**: `dataset_competencias.csv`
- **Ubicación**: `data/dataset_competencias.csv`
- **Tipo**: Dataset estructurado para clasificación multi-etiqueta (multilabel classification)
- **Propósito**: Entrenar un modelo de Machine Learning para identificar competencias laborales basadas en descripciones de CV y talleres de participantes

### 1.2 Estructura del Dataset
- **Total de registros**: 75 instancias de entrenamiento
- **Columnas**: 3 columnas principales
  - `cv_texto`: Texto descriptivo de las experiencias y responsabilidades laborales
  - `talleres`: Temas o cursos relacionados (puede contener múltiples valores separados por comas)
  - `competencias`: Competencias asociadas (clases objetivo, multilabel)

### 1.3 Características del Dataset
- **Formato**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Características principales**:
  - Clasificación multi-etiqueta (un registro puede tener múltiples competencias)
  - Datos textuales (texto libre)
  - Idioma: Español
  - Dominio: Gestión empresarial para PyMEs (Pequeñas y Medianas Empresas)

### 1.4 Competencias Objetivo (24 clases)
El dataset incluye las siguientes competencias laborales:
1. Administración
2. Analítica de Datos
3. Atención al Cliente
4. Calidad
5. Comercio Exterior
6. Compras
7. Comunicación
8. Contabilidad
9. Diseño
10. Finanzas
11. Gestión de Proyectos
12. Legal
13. Logística
14. Mantenimiento
15. Marketing
16. Negociación
17. Ofimática
18. Postventa
19. Producción
20. RRHH
21. Seguridad e Higiene
22. Soporte Técnico
23. Ventas

### 1.5 Ejemplos de Datos
```csv
cv_texto,talleres,competencias
"Atención al cliente en tienda, manejo de caja...",comunicación,Atención al Cliente
"Gestión de cuentas por pagar y cobrar...",contabilidad,Contabilidad
"Armado de reportes en Excel, tablas dinámicas...",excel,Ofimática
```

---

## 2. PROCESOS DE LIMPIEZA Y PREPARACIÓN DE DATOS

### 2.1 Carga y Validación de Datos
**Archivo**: `app/ml/train.py` (líneas 18-27)

```python
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Validación de columnas requeridas
    for col in ["cv_texto", "talleres", "competencias"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")
    
    # Manejo de valores nulos
    df["cv_texto"] = df["cv_texto"].fillna("").astype(str)
    df["talleres"] = df["talleres"].fillna("").astype(str)
    df["competencias"] = df["competencias"].fillna("").astype(str)
    
    return df
```

**Pasos realizados**:
1. Lectura del CSV con `pandas`
2. Validación de existencia de columnas requeridas
3. Manejo de valores nulos: conversión a string vacío
4. Normalización de tipos de datos

### 2.2 Construcción de Features de Texto
**Archivo**: `app/ml/train.py` (líneas 29-37)

```python
def build_text(row) -> str:
    """
    Combina CV + talleres en un solo texto para mejorar la señal.
    Formato: '...texto del cv...' + ' topic:python topic:sql '
    """
    cv = row["cv_texto"].strip().lower()
    talleres = [t.strip().lower() for t in row["talleres"].split(",") if t.strip()]
    taller_tokens = " ".join([f"topic:{t}" for t in talleres])
    return f"{cv} {taller_tokens}"
```

**Transformaciones aplicadas**:
1. **Normalización de texto**: Convertir a minúsculas (`.lower()`)
2. **Eliminación de espacios**: Uso de `.strip()` para limpiar bordes
3. **Tokenización de talleres**: Separar por comas y agregar prefijo `topic:` para distinguir de texto libre
4. **Concatenación**: Combinar texto de CV con tokens de talleres para enriquecer el contexto

### 2.3 Preparación de Etiquetas Multi-etiqueta
**Archivo**: `app/ml/train.py` (líneas 46-49)

```python
# Etiquetas multilabel
y_list = [ 
    [c.strip() for c in comps.split(",") if c.strip()] 
    for comps in df["competencias"].tolist() 
]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_list)
```

**Pasos realizados**:
1. **Parsing de múltiples etiquetas**: Separar competencias por comas
2. **Limpiar y normalizar**: Usar `.strip()` para eliminar espacios
3. **Binarización**: Usar `MultiLabelBinarizer` de scikit-learn para convertir etiquetas categóricas a formato binario (one-hot encoding)
4. Resultado: Matriz binaria donde cada fila es una instancia y cada columna es una competencia (1 si aplica, 0 si no)

### 2.4 División Train/Test
**Archivo**: `app/ml/train.py` (línea 52)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_text, Y, 
    test_size=0.2, 
    random_state=42
)
```

**Configuración**:
- **Proporción**: 80% entrenamiento, 20% prueba
- **Semilla aleatoria**: `random_state=42` para reproducibilidad
- **Balanceo**: El modelo usa `class_weight="balanced"` para manejar desbalance de clases

---

## 3. MODELO DE MACHINE LEARNING

### 3.1 Arquitectura del Modelo
**Tipo**: Pipeline de clasificación multi-etiqueta

```python
pipeline = Pipeline([
    # Feature Engineering
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),      # Unigramas + bigramas
        min_df=1,                # Mínimo 1 documento
        max_df=0.95              # Máximo 95% de documentos
    )),
    
    # Clasificador
    ("clf", OneVsRestClassifier(
        LogisticRegression(
            max_iter=300, 
            class_weight="balanced"
        )
    ))
])
```

### 3.2 Componentes del Pipeline

#### 3.2.1 TfidfVectorizer (Extracción de Features)
- **TF-IDF**: Term Frequency - Inverse Document Frequency
- **ngram_range**: (1,2) captura unigramas ("python", "sql") y bigramas ("análisis de datos")
- **min_df**: 1 (términos que aparecen en al menos 1 documento)
- **max_df**: 0.95 (excluye términos que aparecen en más del 95% de los documentos - demasiado comunes)
- **lowercase**: Normaliza a minúsculas

**Propósito**: Convertir texto libre en vectores numéricos que el modelo puede procesar.

#### 3.2.2 OneVsRestClassifier + LogisticRegression
- **OneVsRestClassifier**: Estrategia uno-contra-todos para clasificación multi-etiqueta
  - Entrena un clasificador binario por cada competencia
  - Cada clasificador predice si esa competencia aplica o no
- **LogisticRegression**:
  - Algoritmo de regresión logística
  - **max_iter**: 300 iteraciones máximas
  - **class_weight**: "balanced" para manejar clases desbalanceadas

**Ventajas de esta arquitectura**:
- Permite clasificación multi-etiqueta (una persona puede tener múltiples competencias)
- Balancea clases automáticamente
- Interpretable: probabilidades por competencia

### 3.3 Proceso de Entrenamiento
```python
# 1. Entrenar
pipeline.fit(X_train, y_train)

# 2. Evaluar
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

# 3. Guardar
artifact = {
    "pipeline": pipeline, 
    "classes": target_names  # Nombres de las competencias
}
joblib.dump(artifact, model_path)
```

### 3.4 Guardado del Modelo
- **Formato**: Archivo `.joblib` serializado con joblib
- **Ubicación**: `models/pipeline_competencias.joblib`
- **Contenido**: Pipeline completo + nombres de clases (competencias)
- **Ventaja**: Carga rápida y reutilizable en producción

---

## 4. INFERENCIA Y PRODUCCIÓN

### 4.1 Carga del Modelo
**Archivo**: `app/ml/model_loader.py`

```python
@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    return pipeline, list(classes)
```

- **Lazy loading**: Carga perezosa (solo cuando se necesita)
- **Cache**: `@lru_cache` para no cargar múltiples veces
- **Componentes**: Pipeline + lista de clases

### 4.2 Predicción en el Servicio
**Archivo**: `app/services/analysis_service.py` (líneas 94-113)

```python
def _predict_with_ml(cv_text: str | None, talleres: List[Dict[str, Any]] | None):
    pipeline, classes = load_model()
    text = _build_text_for_model(cv_text, talleres)
    proba = pipeline.predict_proba([text])[0]
    
    threshold = 0.35  # Umbral configurable
    results = []
    for cls, p in zip(classes, proba):
        if p >= threshold:
            results.append({
                "competencia": cls, 
                "nivel": round(p * 100.0, 1), 
                "confianza": 0.9, 
                "fuente": ["ml"]
            })
    
    results.sort(key=lambda x: -x["nivel"])
    return results
```

**Características**:
- Construye texto de entrada combinando CV + talleres
- Obtiene probabilidades por competencia
- Filtra por umbral (0.35 por defecto)
- Retorna competencias con nivel de confianza

### 4.3 Sistema de Fallback
Si el modelo ML no está disponible o falla, el sistema usa keywords:

**Archivo**: `app/services/job_service.py` (líneas 136-148)

```python
KEYWORDS_MAP = {
    "atención al cliente": "Atención al Cliente",
    "excel": "Ofimática",
    "python": "Programación",
    # ... más keywords
}
```

**Flujo híbrido**:
1. Intentar con ML
2. Si falla, usar keywords basadas en diccionario
3. Combinar resultados si ambos funcionan

---

## 5. MÉTRICAS Y EVALUACIÓN

### 5.1 Métricas Reportadas
Al final del entrenamiento se genera un `classification_report` que incluye:
- **Precision**: Exactitud de predicciones positivas
- **Recall**: Cobertura de instancias positivas
- **F1-Score**: Media armónica de precision y recall
- **Support**: Número de instancias por clase

### 5.2 Umbral de Predicción
- **Configuración**: Variable de entorno `ML_THRESHOLD` (default: 0.35)
- **Interpretación**: Probabilidad mínima para considerar una competencia como detectada
- **Ajuste**: Puede ajustarse según necesidades de negocio

### 5.3 Rango de Predictibilidad
- **Mínimo**: 0.0 (no aplica)
- **Máximo**: 1.0 (certeza total)
- **Normalización**: Convertido a porcentaje (0-100) en producción

---

## 6. STACK TECNOLÓGICO

### 6.1 Biblioteca Principal
- **scikit-learn**: Framework de Machine Learning en Python
  - Clasificadores, preprocesamiento, pipelines

### 6.2 Tecnologías de Soporte
- **pandas**: Manipulación de datos
- **joblib**: Serialización de modelos (equivalente de pickle optimizado para NumPy)
- **TfidfVectorizer**: Vectorización de texto
- **FastAPI**: API de producción para servir el modelo

### 6.3 Algoritmos Utilizados
1. **TF-IDF**: Extracción de características de texto
2. **One-vs-Rest**: Estrategia multi-clase/multi-etiqueta
3. **Logistic Regression**: Clasificador binario subyacente

---

## 7. VENTAJAS DEL DISEÑO

### 7.1 Escalabilidad
- Pipeline reutilizable
- Carga perezosa del modelo
- Cache para inferencias repetidas

### 7.2 Robustez
- Sistema de fallback con keywords
- Manejo de valores nulos
- Validación de datos de entrada

### 7.3 Interpretabilidad
- Probabilidades por competencia
- Umbral configurable
- Fuente de cada competencia (ML vs keywords)

### 7.4 Flexibilidad
- Soporta texto de CV + talleres
- Multi-etiqueta (múltiples competencias por persona)
- Configurable por variables de entorno

---

## 8. ARGUMENTOS CLAVE PARA EVALUACIÓN

### 8.1 Sobre el Dataset
- Dataset real y estructurado para PyMEs
- 75 instancias de entrenamiento
- 24 competencias laborales relevantes
- Multi-etiqueta (caso real de industria)

### 8.2 Sobre Preparación de Datos
- Limpieza robusta (manejo de nulos, normalización)
- Feature engineering (TF-IDF con n-gramas)
- Validación de integridad de datos
- División train/test controlada (80/20, random_state=42)

### 8.3 Sobre el Modelo
- Pipeline completo de scikit-learn (best practices)
- Optimización para clases desbalanceadas (`class_weight="balanced"`)
- Técnicas probadas (TF-IDF + Logistic Regression)
- Arquitectura One-vs-Rest para multi-etiqueta
- Guardado serializado para producción

### 8.4 Sobre Producción
- Sistema híbrido (ML + fallback)
- API REST con FastAPI
- Cache de modelo para performance
- Umbral configurable
- Respuestas estructuradas con métricas de confianza

### 8.5 Impacto del Proyecto
- Automatiza identificación de competencias laborales
- Apoya decisión en PyMEs
- Modelo entrenado y desplegado en producción
- Integración con sistema existente (API REST)

---

## 9. PUNTOS DESTACABLES PARA DEFENSA

1. **Problemática real**: Soluciona necesidad de clasificación de competencias en PyMEs
2. **Metodología sólida**: Proceso completo de ML (limpieza → entrenamiento → despliegue)
3. **Código profesional**: Modular, documentado, con manejo de errores
4. **Producción-ready**: API REST, fallbacks, configuración por variables
5. **Técnicas aplicadas**: TF-IDF, multi-label classification, pipeline de scikit-learn
6. **Dataset representativo**: 24 competencias de PyMEs
7. **Sistema híbrido**: Combina ML con reglas basadas en keywords para robustez
8. **Resultados interpretables**: Probabilidades y niveles de confianza por competencia

---

**Con esta documentación puedes responder preguntas sobre:**
- Origen, estructura y características del dataset
- Procesos de limpieza y preparación (validación, normalización, vectorización)
- Arquitectura del modelo (pipeline, algoritmos, estrategias)
- Proceso de entrenamiento y evaluación
- Sistema de inferencia en producción
- Stack tecnológico utilizado
- Métricas y umbrales
- Ventajas del diseño y decisiones técnicas

