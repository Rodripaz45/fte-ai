# Resumen Ejecutivo - Proyecto de IA: FTE-AI

## DATASET

- **Archivo**: `dataset_competencias.csv`
- **Registros**: 75 instancias de entrenamiento
- **Estructura**: 3 columnas (cv_texto, talleres, competencias)
- **Tipo**: Clasificación multi-etiqueta (un registro puede tener múltiples competencias)
- **Dominio**: Competencias laborales para PyMEs
- **Competencias**: 24 categorías (Atención al Cliente, Ventas, Contabilidad, Logística, etc.)

## LIMPIEZA Y PREPARACIÓN DE DATOS

### 1. Carga y Validación
- Lectura con pandas
- Validación de columnas requeridas
- Manejo de valores nulos (conversión a string vacío)

### 2. Feature Engineering de Texto
```python
# Normalización: texto a minúsculas
# Tokenización: talleres separados por comas
# Enriquecimiento: agregar prefijo "topic:" a talleres
# Concatenación: CV + talleres como tokens
```

Ejemplo: `"Experiencia en Excel y Python"` → `"experiencia en excel y python topic:excel topic:python"`

### 3. Preparación de Etiquetas
- Parsing de múltiples competencias (separadas por comas)
- Binarización con `MultiLabelBinarizer` (one-hot encoding)
- Resultado: Matriz binaria donde cada fila = instancia, cada columna = competencia

### 4. División Train/Test
- **Proporción**: 80% entrenamiento, 20% prueba
- **Semilla**: `random_state=42` para reproducibilidad

## MODELO DE MACHINE LEARNING

### Arquitectura
```python
Pipeline([
    TfidfVectorizer(ngram_range=(1,2)),  # Extracción de features
    OneVsRestClassifier(
        LogisticRegression(class_weight="balanced")
    )
])
```

### Componentes

**TfidfVectorizer**:
- Convierte texto a vectores numéricos
- Captura palabras individuales y frases (ngramas 1-2)
- Excluye términos demasiado comunes (max_df=0.95)

**OneVsRestClassifier + LogisticRegression**:
- Estrategia uno-contra-todos para multi-etiqueta
- Entrena un clasificador por cada competencia
- Balance de clases automático

### Entrenamiento
1. Aplicar transformaciones a texto de CV + talleres
2. Entrenar pipeline con datos de entrenamiento
3. Evaluar con conjunto de prueba (classification_report)
4. Guardar modelo serializado (.joblib)

## PRODUCCIÓN

### Carga del Modelo
- Lazy loading (carga solo cuando se necesita)
- Cache con `@lru_cache`
- Deserialización con joblib

### Inferencia
```python
# Construir texto de entrada
texto = cv + " topic:taller1 topic:taller2"

# Obtener probabilidades
proba = model.predict_proba([texto])[0]

# Filtrar por umbral (0.35)
competencias = [c for c, p in zip(classes, proba) if p >= 0.35]

# Retornar con nivel de confianza
results = [{competencia: c, nivel: p*100} for c, p in competencias]
```

### Sistema Híbrido
- Primero intenta con ML
- Fallback con keywords si ML falla
- Puede combinar ambos

## TECNOLOGÍAS

- **scikit-learn**: Pipeline de ML
- **pandas**: Manipulación de datos
- **joblib**: Serialización de modelos
- **TF-IDF**: Vectorización de texto
- **FastAPI**: API REST para producción

## MÉTRICAS

- **Umbral**: 0.35 (configurable por variable de entorno)
- **Evaluación**: classification_report (precision, recall, F1-score)
- **Normalización**: Probabilidades convertidas a porcentaje (0-100)

## PUNTOS DESTACABLES

1. ✅ **Problemática real**: Identificación automática de competencias laborales
2. ✅ **Proceso completo**: Limpieza → Entrenamiento → Producción
3. ✅ **Código profesional**: Modular, documentado, con fallbacks
4. ✅ **Producción-ready**: API REST, configuración por variables
5. ✅ **Técnicas sólidas**: TF-IDF + Logistic Regression
6. ✅ **Sistema robusto**: Fallback con keywords
7. ✅ **Multi-etiqueta**: Caso real de industria

## RESPUESTAS RÁPIDAS

**¿De dónde viene el dataset?**
- 75 registros estructurados de CV y talleres de participantes
- 24 competencias objetivo para PyMEs

**¿Qué limpieza se hizo?**
- Validación de columnas, manejo de nulos, normalización a minúsculas
- Tokenización de talleres, enriquecimiento con prefijo "topic:"
- Binarización para clasificación multi-etiqueta

**¿Qué modelo se usó?**
- Pipeline: TF-IDF + OneVsRestClassifier(LogisticRegression)
- Arquitectura: Uno-contra-todos para multi-etiqueta
- Balance: class_weight="balanced" para manejar clases desbalanceadas

**¿Cómo funciona en producción?**
- API REST con FastAPI
- Predicción con probabilidades por competencia
- Umbral de 0.35 para filtrar predicciones
- Fallback con keywords si ML falla

**¿Por qué este modelo?**
- TF-IDF captura términos relevantes (palabras + frases)
- Logistic Regression es interpretable (probabilidades)
- OneVsRest permite multi-etiqueta (múltiples competencias)
- Balanceo automático de clases

