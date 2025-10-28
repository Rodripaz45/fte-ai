# Preguntas Frecuentes - Defensa del Proyecto FTE-AI

---

## 1. ¿Por qué elegiste Machine Learning para este problema?

**Respuesta**:
- El problema de clasificación de competencias laborales es complejo y no puede resolverse con reglas simples, ya que las descripciones son en texto libre
- Se necesitaba un modelo que aprenda patrones de texto (sinónimos, contexto, relaciones semánticas)
- El volumen de datos (75 instancias) es suficiente para entrenar un modelo de clasificación multi-etiqueta
- Permite escalar a nuevas competencias sin reescribir reglas manualmente
- El sistema híbrido (ML + keywords) combina aprendizaje automático con conocimiento experto

---

## 2. ¿Por qué usaste Logistic Regression y no otro algoritmo?

**Respuesta**:
- **Interpretabilidad**: Devuelve probabilidades directas, no solo la clase predicha
- **Multi-etiqueta**: OneVsRest con LogisticRegression es una estrategia probada para clasificación multi-etiqueta
- **Performance**: Rápido para entrenar y predecir, importante para producción
- **Balance de clases**: `class_weight="balanced"` maneja automáticamente datasets desbalanceados
- **Tamaño del dataset**: Para 75 instancias, un modelo complejo como Random Forest o redes neuronales podría sobreajustar
- **Estabilidad**: Comportamiento predecible y consistente

---

## 3. ¿Por qué TF-IDF y no embeddings (Word2Vec, BERT)?

**Respuesta**:
- **Tamaño del dataset**: 75 instancias es muy pequeño para embeddings profundos o BERT
- **Sobrecarga computacional**: BERT requiere GPU y mucho tiempo de entrenamiento
- **Simplicidad**: TF-IDF es simple, eficiente y funciona bien para este tamaño de dataset
- **Interpretabilidad**: Los pesos TF-IDF son más fáciles de entender que embeddings
- **Reproducibilidad**: TF-IDF con parámetros fijos produce resultados consistentes
- **Producción**: FastAPI con TF-IDF tiene baja latencia y bajo consumo de recursos

**Nota adicional**: Para un proyecto futuro con más datos, se podría migrar a embeddings o transformers

---

## 4. ¿Cómo validaste la calidad del modelo?

**Respuesta**:
- **División train/test**: 80/20 con `random_state=42` para reproducibilidad
- **Métricas**: `classification_report` incluye precision, recall, F1-score por clase
- **Balanceo**: `class_weight="balanced"` para manejar competencias minoritarias
- **Umbral configurable**: Variable `ML_THRESHOLD` (default 0.35) ajustable según necesidades
- **Sistema de fallback**: Si ML falla, usa keywords basadas en conocimiento experto
- **Evaluación continua**: En producción se podría agregar logging y A/B testing

---

## 5. ¿El dataset es suficiente para entrenar un modelo?

**Respuesta**:
- **75 instancias** es mínimo pero funcional para 24 clases
- **Multi-etiqueta**: Aunque el dataset sea pequeño, cada instancia aporta señal para múltiples competencias
- **Feature engineering**: El uso de TF-IDF maximiza la señal de texto disponible
- **Sistema híbrido**: ML complementado con keywords asegura robustez
- **Aprendizaje incremental**: El modelo puede mejorarse agregando más datos sin re-entrenar desde cero
- **Enfoque pragmático**: Para un MVP/prototipo, 75 instancias validan el concepto

**Nota**: En producción se recomienda expandir el dataset con más datos

---

## 6. ¿Por qué usaste n-gramas (1,2) en TF-IDF?

**Respuesta**:
- **Unigramas (1)**: Capturan palabras individuales ("python", "excel", "ventas")
- **Bigramas (2)**: Capturan frases y contexto ("análisis de datos", "atención al cliente", "gestión de proyectos")
- **Combinación**: Captura tanto términos simples como frases características del dominio
- **Relevancia en PyMEs**: Frases como "control de inventario" o "conciliaciones bancarias" son muy específicas
- **Balance**: Evita n-gramas de orden superior que podrían ser demasiado específicos y ruidosos

---

## 7. ¿Cómo manejaste el desbalance de clases?

**Respuesta**:
- **class_weight="balanced"**: Logistic Regression ajusta automáticamente los pesos de clases
- **OneVsRest**: Entrena un modelo independiente por competencia, cada uno con sus propios pesos
- **Umbral configurable**: `ML_THRESHOLD` permite ajustar la sensibilidad del modelo
- **Sistema híbrido**: Keywords ayudan a detectar competencias que el ML podría ignorar
- **Evaluación**: `classification_report` muestra métricas por clase para identificar áreas problemáticas

---

## 8. ¿Qué pasa si el modelo predice mal?

**Respuesta**:
- **Sistema de fallback**: Si ML falla o no está disponible, usa diccionario de keywords (66 palabras clave mapeadas a competencias)
- **Umbral conservador**: 0.35 es relativamente conservador, evita falsos positivos
- **Transparencia**: La API retorna la "fuente" de cada competencia (ML vs keywords)
- **Nivel de confianza**: Cada predicción incluye un score que permite validar la confiabilidad
- **Logging**: Se puede agregar logging para monitorear predicciones y detectar errores

---

## 9. ¿Por qué multi-etiqueta y no multi-clase?

**Respuesta**:
- **Caso real**: Una persona puede tener múltiples competencias (contabilidad + excel + atención al cliente)
- **Realidad laboral**: Los perfiles profesionales son diversos, no excluyentes
- **Valor de negocio**: Identificar todas las competencias potenciales es más útil que elegir solo una
- **OneVsRest**: Permite naturalmente este tipo de clasificación
- **Interpretabilidad**: Se obtienen probabilidades por competencia, mostrando un perfil completo

---

## 10. ¿Cuál es la diferencia entre los dos endpoints (profile y job)?

**Respuesta**:
- **`/analyze/profile`**: Analiza el **perfil de una persona** (CV + talleres) para identificar sus competencias
  - Input: CV texto + talleres asistidos
  - Output: Competencias de la persona
  
- **`/analyze/job`**: Analiza la **descripción de un puesto** para identificar competencias requeridas
  - Input: Descripción de puesto/necesidades
  - Output: Competencias necesarias para ese puesto

**Aplicación**: Se pueden combinar ambos para hacer **matching** de personas con puestos

---

## 11. ¿Por qué guardaste el modelo con joblib y no pickle?

**Respuesta**:
- **Joblib**: Optimizado para arrays de NumPy (más eficiente)
- **Compatible con scikit-learn**: Mejor manejo de pipelines complejos
- **Serialización paralela**: Soporte para multiprocessing
- **Estabilidad**: Más confiable para modelos grandes
- **Best practice**: Recomendado por la documentación de scikit-learn
- **Realidad**: Más usado en producción

---

## 12. ¿El modelo se re-entrena automáticamente?

**Respuesta**:
No, el modelo actual es estático:
- Se entrena manualmente con `python app/ml/train.py`
- Se guarda en disco (`models/pipeline_competencias.joblib`)
- La API carga el modelo pre-entrenado (lazy loading)

**Para producción**:
- Se podría implementar re-entrenamiento automático
- Requeriría sistema de versioning de modelos
- Pipeline de CI/CD para entrenamientos programados
- A/B testing de diferentes versiones

---

## 13. ¿Cómo funciona el sistema híbrido (ML + keywords)?

**Respuesta**:
1. **Intento ML**: Se intenta usar el modelo entrenado
2. **Si falla**: Usa diccionario de keywords (66 palabras clave → competencia)
3. **Si ambos funcionan**: Combina resultados, eliminando duplicados
4. **Transparencia**: Marca la "fuente" de cada competencia

**Ventajas**:
- Robustez: Siempre hay respuesta, incluso si ML falla
- Interpretabilidad: Se sabe si viene de ML o keywords
- Flexibilidad: Se puede ajustar el diccionario de keywords fácilmente

---

## 14. ¿Por qué la normalización a minúsculas?

**Respuesta**:
- **Consistencia**: Evita duplicados ("Python" vs "python")
- **Homogeneización**: En español, las mayúsculas son frecuentes al inicio de oraciones
- **Matching**: Mejor para búsqueda y comparación de términos
- **TF-IDF**: Es estándar convertir a minúsculas antes de vectorización
- **Eficiencia**: Reduce el vocabulario (menos tokens únicos)

---

## 15. ¿Qué mejoras podrías implementar a futuro?

**Respuesta**:
1. **Más datos**: Expandir dataset para mejorar generalización
2. **Word embeddings**: Usar Word2Vec o FastText para capturar sinónimos
3. **Transformers**: Considerar modelos como BERT para español (si hay suficiente data)
4. **Aprendizaje incremental**: Re-entrenar con nuevos datos periódicamente
5. **Métricas de negocio**: Tracking de accuracy en casos reales
6. **A/B testing**: Comparar diferentes versiones del modelo
7. **Feedback loop**: Incorporar correcciones de usuarios para mejorar el modelo
8. **API de versioning**: Mantener múltiples versiones del modelo en producción

---

## 16. ¿Cómo evalúas el impacto del proyecto?

**Respuesta**:
- **Automatización**: Reduce tiempo de análisis manual de CV
- **Escalabilidad**: Puede analizar cientos de perfiles en minutos
- **Precisión**: Modelo + keywords reduce errores vs análisis manual
- **Decisiones informadas**: Provee datos cuantitativos (probabilidades) para tomar decisiones
- **Casos de uso**:
  - Matching de candidatos con puestos
  - Identificación de competencias faltantes
  - Personalización de ofertas de trabajo
  - Análisis de brechas de habilidades en equipos

---

## 17. ¿Por qué split 80/20 y no 70/30?

**Respuesta**:
- **Balance**: 20% de test es suficiente para evaluación estadística confiable
- **Data limitado**: Con solo 75 instancias, 80% da ~60 para entrenar (necesario para 24 clases)
- **Estándar**: 80/20 es convención común en ML (hay variaciones 70/30 o 90/10 según contexto)
- **Reproducibilidad**: `random_state=42` asegura resultados consistentes

**Nota**: Si hubiera más datos (p.ej. 1000+), se podría considerar 70/30

---

## 18. ¿Cómo seleccionaste los parámetros del modelo?

**Respuesta**:
- **TF-IDF**: 
  - `ngram_range=(1,2)`: Convención estándar, balance entre simplicidad y contexto
  - `max_df=0.95`: Excluye palabras demasiado comunes (stopwords implícitas)
  - `min_df=1`: Incluye todos los términos (dataset pequeño)
  
- **Logistic Regression**:
  - `max_iter=300`: Suficiente para convergencia sin ser excesivo
  - `class_weight="balanced"`: Maneja desbalance sin necesidad de oversampling
  - `OneVsRest`: Estándar para multi-etiqueta en scikit-learn

**Ventaja**: Parámetros no requieren fine-tuning extensivo para este tamaño de dataset

---

## 19. ¿Qué métricas reportaste?

**Respuesta**:
El `classification_report` de scikit-learn reporta:
- **Precision**: De las predicciones positivas, ¿cuántas son correctas?
- **Recall**: De los casos positivos reales, ¿cuántos capturó el modelo?
- **F1-Score**: Media armónica de precision y recall (balance)
- **Support**: Número de instancias por clase (útil para identificar clases minoritarias)

**Por defecto**: Reporte por clase (micro, macro, weighted averages)

---

## 20. ¿Cómo explicarías el proyecto a alguien no técnico?

**Respuesta**:
"Este sistema analiza currículums y talleres de personas para identificar automáticamente sus habilidades profesionales. 

Por ejemplo, si alguien menciona que sabe Excel, Python y atención al cliente, el sistema identifica las competencias: Analítica de Datos, Programación y Atención al Cliente.

Funciona como un filtro inteligente que lee el texto del CV y asigna probabilidades. No necesita reglas manuales - aprende de ejemplos previos y puede procesar cientos de CVs en minutos.

Es útil para PyMEs que necesitan identificar rápidamente qué personas tienen las habilidades que buscan."

