# Diagramas y Flujos del Proyecto FTE-AI

## 1. FLUJO GENERAL DEL SISTEMA

```
┌─────────────────┐
│   Dataset CSV   │
│  75 registros   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   LIMPIEZA Y PREPARACIÓN           │
│  - Validación de columnas          │
│  - Manejo de nulos                 │
│  - Normalización de texto          │
│  - Construcción de features        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   TRAIN/TEST SPLIT (80/20)         │
│  - 60 instancias: entrenamiento     │
│  - 15 instancias: prueba           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   ENTRENAMIENTO DEL MODELO          │
│  - TF-IDF Vectorization            │
│  - OneVsRest + LogisticRegression  │
│  - Guardado del modelo             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   SERVIDOR API (FastAPI)            │
│  - Carga lazy del modelo           │
│  - Endpoint: /analyze/profile      │
│  - Endpoint: /analyze/job          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   INFERENCIA EN PRODUCCIÓN          │
│  - Input: CV + talleres              │
│  - Output: Competencias + scores    │
└─────────────────────────────────────┘
```

---

## 2. PROCESO DE PREPARACIÓN DE DATOS

```
DATASET ORIGINAL (CSV)
├─ cv_texto: "Gestión de cuentas por pagar..."
├─ talleres: "contabilidad"
└─ competencias: "Contabilidad"

         │
         ▼

NORMALIZACIÓN
├─ Minúsculas: "gestión de cuentas..."
├─ Trim de espacios
└─ Manejo de nulos

         │
         ▼

FEATURE ENGINEERING
├─ CV: "gestión de cuentas por pagar..."
├─ Talleres tokenizados: " topic:contabilidad "
└─ Texto combinado: "gestión... topic:contabilidad"

         │
         ▼

BINARIZACIÓN (MultiLabelBinarizer)
Comp.1  Comp.2  Comp.3  ...  Comp.24
  0       1       0     ...     0
  ↑              ↑
└─ no tiene    tiene
```

---

## 3. PIPELINE DEL MODELO

```
INPUT: Texto combinado
"gestión de cuentas... topic:contabilidad"

         │
         ▼
┌─────────────────────────────────┐
│   TF-IDF VECTORIZER            │
│                                 │
│   Entrada: String               │
│   Salida: Vector [0.2, 0.5,    │
│                 0.1, 0.8, ...] │
│                                 │
│   Características:              │
│   - Unigramas + Bigramas       │
│   - Ponderación TF-IDF          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   MULTILABEL BINARIZER          │
│   (24 competencias)             │
│                                 │
│   [0, 1, 0, 0, ..., 0]         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   ONE-VS-REST CLASSIFIER        │
│                                 │
│   24 clasificadores binarios:   │
│   ┌──────────────────────────┐  │
│   │ Clf 1: Atención Cliente? │  │
│   │ Clf 2: Ventas?          │  │
│   │ Clf 3: Contabilidad?    │  │
│   │ ...                     │  │
│   │ Clf 24: Legal?         │  │
│   └──────────────────────────┘  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   PROBABILIDADES                 │
│                                 │
│   [0.15, 0.42, 0.88, ..., 0.03]│
│   ↑         ↑     ↑             │
│   Relevancia proporcional       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   FILTRADO POR UMBRAL (0.35)    │
│                                 │
│   [N, N, Y, N, ..., N]          │
│   (N = bajo umbral, Y = alto)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   RESULTADO FINAL               │
│                                 │
│   Competencias detectadas:      │
│   - Contabilidad (0.88)         │
│   - Finanzas (0.42)            │
└─────────────────────────────────┘
```

---

## 4. ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT (POSTMAN/cURL)            │
└────────────────────────┬────────────────────────────┘
                         │
                         │ HTTP POST /analyze/profile
                         ▼
┌─────────────────────────────────────────────────────┐
│              FASTAPI SERVER                        │
│  ┌──────────────────────────────────────────────┐  │
│  │   Router: analyze_routes.py                   │  │
│  │   - Recibe payload                           │  │
│  │   - Valida con Pydantic                       │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   Service: analysis_service.py              │  │
│  │   - Extrae CV y talleres                    │  │
│  │   - Construye texto combinado               │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   Model Loader: model_loader.py             │  │
│  │   (Cache con @lru_cache)                    │  │
│  │   - Carga pipeline + clases                 │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   ML Pipeline: pipeline.joblib               │  │
│  │   - TF-IDF + OneVsRest(LogReg)              │  │
│  │   - Predicción de probabilidades            │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   Fallback: KEYWORDS_MAP (job_service.py)   │  │
│  │   - Si ML falla, usa diccionario de 66      │  │
│  │     keywords                                 │  │
│  └────────────┬─────────────────────────────────┘  │
│               │                                     │
│               ▼                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   Response: JSON con competencias           │  │
│  │   {                                          │  │
│  │     "competencias": [...]                   │  │
│  │     "meta": {...}                           │  │
│  │   }                                          │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   CLIENT recibe JSON                │
└─────────────────────────────────────────────────────┘
```

---

## 5. FLUJO DE ENTRENAMIENTO

```
PROCESO DE ENTRENAMIENTO
════════════════════════

1️⃣  CARGA DE DATOS
    ├─ Leer CSV (75 filas)
    ├─ Validar columnas
    └─ Filtrar nulos
    ┌─────────────────────────┐
    │ Dataset limpio          │
    └─────────────────────────┘

2️⃣  CONSTRUCCIÓN DE FEATURES
    ├─ build_text(): Combina CV + talleres
    ├─ Normaliza a minúsculas
    └─ Agrega prefijo "topic:" a talleres
    ┌─────────────────────────┐
    │ X_text: Lista de textos │
    └─────────────────────────┘

3️⃣  PREPARACIÓN DE ETIQUETAS
    ├─ Split competencias por comas
    ├─ MultiLabelBinarizer
    └─ Matriz binaria Y
    ┌─────────────────────────┐
    │ Y: Matriz (75 x 24)     │
    └─────────────────────────┘

4️⃣  TRAIN/TEST SPLIT
    ├─ 80% train (60 instancias)
    ├─ 20% test (15 instancias)
    └─ random_state=42
    ┌────────────┬────────────┐
    │   X_train  │   X_test   │
    │   y_train  │   y_test   │
    └────────────┴────────────┘

5️⃣  ENTRENAMIENTO
    ├─ Pipeline.fit(X_train, y_train)
    ├─ TF-IDF + LogisticRegression
    └─ 24 clasificadores binarios
    ┌─────────────────────────┐
    │ Modelo entrenado        │
    └─────────────────────────┘

6️⃣  EVALUACIÓN
    ├─ y_pred = pipeline.predict(X_test)
    ├─ classification_report
    └─ Matriz de confusión
    ┌─────────────────────────┐
    │ Métricas por competencia │
    └─────────────────────────┘

7️⃣  SERIALIZACIÓN
    ├─ artifact = {pipeline, classes}
    ├─ joblib.dump(artifact)
    └─ Guardar en models/
    ┌─────────────────────────┐
    │ pipeline.joblib         │
    └─────────────────────────┘
```

---

## 6. EJEMPLO DE PREDICCIÓN

```
INPUT DEL CLIENTE
═════════════════
{
  "participanteId": "P001-2024",
  "cvTexto": "Experiencia en Excel y análisis de datos",
  "talleres": [
    {"tema": "Excel", "asistencia_pct": 0.85}
  ]
}

PROCESAMIENTO INTERNO
═════════════════════
1. Construir texto:
   "experiencia en excel y análisis de datos topic:excel"

2. TF-IDF vectorization:
   [0.12, 0.45, 0.31, 0.78, ..., 0.02]
   └─ Valores TF-IDF por término

3. Predicción ML:
   Probabilidades por competencia:
   Atención al Cliente:  0.15
   Ofimática:            0.82  ← Alta
   Analítica de Datos:   0.65  ← Alta
   Ventas:               0.23
   ...

4. Filtrar por umbral (0.35):
   ✅ Ofimática: 0.82
   ✅ Analítica de Datos: 0.65

OUTPUT AL CLIENTE
═════════════════
{
  "participanteId": "P001-2024",
  "competencias": [
    {
      "competencia": "Ofimática",
      "nivel": 82.0,
      "confianza": 0.9,
      "fuente": ["ml"]
    },
    {
      "competencia": "Analítica de Datos",
      "nivel": 65.0,
      "confianza": 0.9,
      "fuente": ["ml"]
    }
  ],
  "meta": {
    "mode": "ml",
    "threshold": "0.35"
  }
}
```

---

## 7. SISTEMA HÍBRIDO (ML + Keywords)

```
NUEVA PETICIÓN
══════════════

┌─────────────────────────────────────┐
│  Intentar con ML                    │
└──────────────┬──────────────────────┘
               │
         ┌─────┴─────┐
         │           │
    ✓ Funciona  ✗ Falla
         │           │
         ▼           ▼
┌──────────────┐ ┌──────────────┐
│ RESULTADO ML │ │   FALLBACK   │
│              │ │   KEYWORDS   │
│  [0.82]      │ │              │
│  [0.65]      │ │  Diccionario  │
│              │ │  66 keywords │
└──────────────┘ └──────────────┘
         │           │
         └─────┬─────┘
               │
         ┌─────┴─────┐
         │           │
    ¿Combinar?   ¿Solo uno?
         │           │
         ▼           ▼
┌──────────────┐ ┌──────────────┐
│ COMBINACIÓN  │ │  RESULTADO   │
│ ML + Keywords│ │   ÚNICO      │
│ (sin dups)   │ │              │
└──────────────┘ └──────────────┘
         │           │
         └─────┬─────┘
               │
               ▼
        ┌──────────────┐
        │ JSON OUTPUT  │
        │ (con fuente) │
        └──────────────┘
```

---

## 8. MÉTRICAS Y EVALUACIÓN

```
CLASSIFICATION REPORT
═════════════════════

                    precision    recall  f1-score   support

Atención Cliente       0.XX      0.XX     0.XX         X
Ventas                 0.XX      0.XX     0.XX         X
Contabilidad           0.XX      0.XX     0.XX         X
...

micro avg              0.XX      0.XX     0.XX        XXX
macro avg              0.XX      0.XX     0.XX         X
weighted avg           0.XX      0.XX     0.XX        XXX

INTERPRETACIÓN
══════════════
- precision: De las predicciones positivas, cuántas son correctas
- recall: De los casos reales positivos, cuántos detectó
- f1-score: Balance entre precision y recall
- support: Número de instancias en test set por clase
```

---

## 9. COMPARACIÓN: ENFOQUE MANUAL vs ML

```
ENFOQUE MANUAL (Reglas hardcoded)
══════════════════════════════════
Input: "Experiencia en Excel y análisis de datos"

↓

if "excel" in texto:
    agregar_competencia("Excel")
if "análisis" in texto and "datos" in texto:
    agregar_competencia("Analítica de Datos")

↓

Output: ["Excel", "Analítica de Datos"]

LIMITACIONES:
- No captura sinónimos
- Reglas rígidas
- No aprende de contexto
- Difícil de mantener


ENFOQUE ML (Aprendizaje)
═════════════════════════
Input: "Experiencia en Excel y análisis de datos"

↓

TF-IDF → Vectores numéricos
OneVsRest → 24 clasificadores
Predict_proba → [0.02, 0.82, 0.65, ...]

↓

Filtrado por umbral (0.35)
Ordenamiento por probabilidad

↓

Output: [
    {competencia: "Analítica de Datos", nivel: 0.65},
    {competencia: "Ofimática", nivel: 0.82}
]

VENTAJAS:
- Aprende patrones automáticamente
- Captura sinónimos y contexto
- Probabilidades cuantificables
- Se adapta a nuevos datos
```

---

## 10. RESUMEN DE COMPONENTES

```
COMPONENTES DEL PROYECTO
═════════════════════════

📊 DATASET
   ├─ 75 instancias
   ├─ 24 competencias
   └─ 3 columnas (cv_texto, talleres, competencias)

🧹 PREPROCESAMIENTO
   ├─ Normalización de texto
   ├─ Feature engineering
   ├─ Binarización multi-label
   └─ Train/test split (80/20)

🤖 MODELO ML
   ├─ TfidfVectorizer (ngrams 1-2)
   ├─ OneVsRestClassifier
   ├─ LogisticRegression (balanced)
   └─ Pipeline completo

💾 PERSISTENCIA
   ├─ joblib serialization
   ├─ Cache con @lru_cache
   └─ Lazy loading

🌐 API
   ├─ FastAPI server
   ├─ Endpoints REST
   ├─ Validación Pydantic
   └─ JSON responses

🔄 FALLBACK
   ├─ Keywords mapping (66)
   ├─ Sistema híbrido
   └─ Transparencia de fuentes

⚙️ CONFIGURACIÓN
   ├─ Variables de entorno
   ├─ Umbral configurable
   └─ Logs y métricas
```


