from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple
from app.ml.model_loader import load_model

ML_THRESHOLD_ENV = os.getenv("ML_THRESHOLD")

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

def _get_threshold(metadata: Dict[str, Any]) -> float:
    if ML_THRESHOLD_ENV is not None:
        try:
            return float(ML_THRESHOLD_ENV)
        except ValueError:
            pass
    return float(metadata.get("best_threshold", metadata.get("threshold", 0.35)))

def _predict_ml(texto: str, top_k: int) -> List[Dict[str, Any]]:
    pipe, classes, metadata = load_model()
    if pipe is None:
        return []
    text = (texto or "").strip().lower()
    try:
        probs = pipe.predict_proba([text])[0]
    except Exception:
        import numpy as np
        logits = pipe.decision_function([text])[0]
        probs = 1 / (1 + np.exp(-logits))
    threshold = _get_threshold(metadata)
    min_prob_floor = float(os.getenv("ML_MIN_PROB_FLOOR", "0.25"))

    # 1) Por encima del umbral
    scored: List[Tuple[str, float]] = [
        (label, float(p))
        for label, p in zip(classes if classes else [f"Clase_{i}" for i in range(len(probs))], probs)
        if float(p) >= threshold
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    # 2) Si faltan hasta top_k, completar con los mejores restantes sobre el piso
    if top_k and top_k > 0 and len(scored) < top_k:
        chosen = {l for l, _ in scored}
        remaining = [
            (label, float(p))
            for label, p in zip(classes if classes else [f"Clase_{i}" for i in range(len(probs))], probs)
            if label not in chosen
        ]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for label, p in remaining:
            if p < min_prob_floor:
                break
            scored.append((label, p))
            if len(scored) >= top_k:
                break

    return [
        {"competencia": label, "nivel": round(score * 100.0, 1), "confianza": (0.9 if score >= threshold else 0.75), "fuente": ["ml"]}
        for label, score in scored
    ]

def _predict_keywords(texto: str, top_k: int) -> List[Dict[str, Any]]:
    t = " " + texto.lower() + " "
    counts: Dict[str, float] = {}
    for kw, comp in KEYWORDS_MAP.items():
        if f" {kw} " in t:
            counts[comp] = counts.get(comp, 0.0) + 1.0
    results = [
        {"competencia": comp, "nivel": round(min(0.75, 0.35 + c * 0.2) * 100.0, 1), "confianza": 0.6, "fuente": ["keywords"]}
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
            },
        }
    # 3) Si hay ML, enriquecemos con keywords no duplicadas
    kw_results = _predict_keywords(puesto_texto, top_k)
    names_ml = {c["competencia"] for c in ml_results}
    merged = ml_results + [c for c in kw_results if c["competencia"] not in names_ml]
    merged.sort(key=lambda x: x["nivel"], reverse=True)
    if top_k and top_k > 0:
        merged = merged[:top_k]
    return {
        "competencias": merged,
        "meta": {"mode": "ml+keywords"},
    }
