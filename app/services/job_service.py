from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
import joblib

ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.35"))

# Carga perezosa del pipeline entrenado
_PIPELINE = None
_PIPELINE_CLASSES = None
_PIPELINE_PATH = os.getenv("ML_MODEL_PATH", "models/pipeline_competencias.joblib")

# Diccionario fallback de keywords → competencia (enfocado en PyMEs)
KEYWORDS_MAP = {
    "atención al cliente": "Atención al Cliente",
    "caja": "Atención al Cliente",
    "postventa": "Postventa",
    "reclamos": "Atención al Cliente",
    "ventas": "Ventas",
    "comercial": "Ventas",
    "prospección": "Ventas",
    "negociación": "Negociación",
    "cotizaciones": "Compras",
    "proveedores": "Compras",
    "orden de compra": "Compras",
    "inventario": "Logística",
    "almacén": "Logística",
    "despacho": "Logística",
    "ruteo": "Logística",
    "excel": "Ofimática",
    "power bi": "Analítica de Datos",
    "tablas dinámicas": "Ofimática",
    "contabilidad": "Contabilidad",
    "conciliaciones": "Contabilidad",
    "facturación": "Contabilidad",
    "flujo de caja": "Finanzas",
    "presupuesto": "Finanzas",
    "rrhh": "RRHH",
    "reclutamiento": "RRHH",
    "inducción": "RRHH",
    "planilla de sueldos": "RRHH",
    "calidad": "Calidad",
    "no conformidades": "Calidad",
    "5s": "Calidad",
    "mantenimiento": "Mantenimiento",
    "correctivo": "Mantenimiento",
    "preventivo": "Mantenimiento",
    "producción": "Producción",
    "scrap": "Producción",
    "seguridad e higiene": "Seguridad e Higiene",
    "epi": "Seguridad e Higiene",
    "comercio exterior": "Comercio Exterior",
    "despachante": "Comercio Exterior",
    "diseño": "Diseño",
    "canva": "Diseño",
    "marketing": "Marketing",
    "redes sociales": "Marketing",
    "community": "Marketing",
    "soporte": "Soporte Técnico",
    "ofimática": "Ofimática",
    "administración": "Administración",
    "documentación": "Administración",
    "proyectos": "Gestion de Proyectos",
    "cronograma": "Gestion de Proyectos",
}

def _load_pipeline():
    global _PIPELINE, _PIPELINE_CLASSES
    if _PIPELINE is None and os.path.exists(_PIPELINE_PATH):
        loaded = joblib.load(_PIPELINE_PATH)
        # Si se carga un diccionario, extraer el pipeline y las clases
        if isinstance(loaded, dict):
            # Buscar el objeto que tenga el método predict_proba
            for key, value in loaded.items():
                if hasattr(value, 'predict_proba'):
                    _PIPELINE = value
                    # Intentar obtener las clases
                    if 'classes' in loaded:
                        _PIPELINE_CLASSES = loaded['classes']
                    elif hasattr(_PIPELINE, 'classes_'):
                        _PIPELINE_CLASSES = _PIPELINE.classes_
                    break
            else:
                # Si no se encuentra, usar el primer elemento que sea un pipeline
                _PIPELINE = loaded.get('pipeline') or loaded.get('model') or loaded.get('pipe')
                _PIPELINE_CLASSES = loaded.get('classes')
        else:
            _PIPELINE = loaded
            if hasattr(loaded, 'classes_'):
                _PIPELINE_CLASSES = loaded.classes_
    return _PIPELINE

def _predict_ml(texto: str, top_k: int) -> List[Dict[str, Any]]:
    pipe = _load_pipeline()
    if pipe is None:
        return []
    
    # Validar que el pipeline tenga los métodos necesarios
    if not hasattr(pipe, 'predict_proba'):
        return []
    
    try:
        # El pipeline es OneVsRest(LogReg) sobre TfidfVectorizer
        # Obtenemos probabilidades por clase
        probs = pipe.predict_proba([texto])[0]
        
        # Obtener las clases del diccionario cargado
        global _PIPELINE_CLASSES
        if _PIPELINE_CLASSES is not None:
            labels: List[str] = list(_PIPELINE_CLASSES)
        elif hasattr(pipe, 'classes_'):
            labels = list(pipe.classes_)
        elif hasattr(pipe, 'estimator') and hasattr(pipe.estimator, 'classes_'):
            labels = list(pipe.estimator.classes_)
        else:
            # Si no hay clases definidas, usar índices
            labels = [f"Clase_{i}" for i in range(len(probs))]
        
        scored: List[Tuple[str, float]] = []
        for label, p in zip(labels, probs):
            if p >= ML_THRESHOLD:
                scored.append((label, float(p)))
        # Ordenar de mayor a menor y recortar
        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k and top_k > 0:
            scored = scored[:top_k]

        return [
            {"nombre": label, "nivel": round(score, 4), "fuente": ["ml"]}
            for label, score in scored
        ]
    except Exception as e:
        # Si hay algún error, devolver lista vacía para usar el fallback
        return []

def _predict_keywords(texto: str, top_k: int) -> List[Dict[str, Any]]:
    t = " " + texto.lower() + " "
    counts: Dict[str, float] = {}
    for kw, comp in KEYWORDS_MAP.items():
        if f" {kw} " in t:
            counts[comp] = counts.get(comp, 0.0) + 1.0
    # Normalizamos rudimente (0.35–0.75 para que sea comparable a threshold)
    results = [
        {"nombre": comp, "nivel": min(0.75, 0.35 + c * 0.2), "fuente": ["keywords"]}
        for comp, c in counts.items()
    ]
    results.sort(key=lambda x: x["nivel"], reverse=True)
    return results[:top_k] if top_k and top_k > 0 else results

def analyze_job_requirements(puesto_texto: str, top_k: int = 6) -> Dict[str, Any]:
    # 1) ML
    ml_results = _predict_ml(puesto_texto, top_k)
    # 2) Fallback por keywords si ML no devuelve nada
    if not ml_results:
        kw_results = _predict_keywords(puesto_texto, top_k)
        return {
            "competencias": kw_results,
            "meta": {
                "mode": "keywords" if kw_results else "none",
                "threshold": str(ML_THRESHOLD),
            },
        }
    # 3) Si hay ML, enriquecemos con keywords no duplicadas
    kw_results = _predict_keywords(puesto_texto, top_k)
    names_ml = {c["nombre"] for c in ml_results}
    merged = ml_results + [c for c in kw_results if c["nombre"] not in names_ml]
    # recortar a top_k si hace falta
    merged.sort(key=lambda x: x["nivel"], reverse=True)
    if top_k and top_k > 0:
        merged = merged[:top_k]
    return {
        "competencias": merged,
        "meta": {"mode": "ml+keywords", "threshold": str(ML_THRESHOLD)},
    }
