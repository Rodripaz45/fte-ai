from __future__ import annotations
from collections import defaultdict
from math import log
from typing import Any, Dict, List, Tuple
import os

# ====== Reglas actuales (fallback) ======
TOPIC_TO_COMPETENCIAS: Dict[str, List[Tuple[str, float]]] = {
    "excel": [("Analítica de Datos", 1.0)],
    "python": [("Programación", 1.0), ("Analítica de Datos", 0.8)],
    "sql": [("Analítica de Datos", 1.0)],
    "comunicación": [("Comunicación", 1.0)],
    "liderazgo": [("Liderazgo", 1.0)],
    "power bi": [("Analítica de Datos", 0.9)],
}
COMPETENCIA_SYNONYMS: Dict[str, List[str]] = {
    "Programación": ["programación", "python", "desarrollo", "algoritmos", "scripts"],
    "Analítica de Datos": ["excel", "sql", "reportes", "power bi", "tableau", "dashboards", "etl", "analítica"],
    "Comunicación": ["presentaciones", "comunicación", "oratoria", "storytelling", "escritura"],
    "Liderazgo": ["liderazgo", "gestión de equipos", "coaching", "mentoría", "resolución de conflictos"],
}
W_TALLERES = 0.6
W_CV = 0.4

def _normalize_text(s: str) -> str:
    return s.strip().lower()

def _safe(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def _score_from_talleres(talleres: List[Dict[str, Any]]) -> Dict[str, float]:
    raw = defaultdict(float)
    for t in talleres:
        tema = _normalize_text(t.get("tema", ""))
        asistencia = float(t.get("asistencia_pct", 0.0))
        if tema in TOPIC_TO_COMPETENCIAS and asistencia > 0:
            for comp, weight in TOPIC_TO_COMPETENCIAS[tema]:
                raw[comp] += asistencia * weight * 100.0
    if not raw:
        return {}
    max_score = max(raw.values()) or 1.0
    return {comp: round(v / max_score * 100.0, 1) for comp, v in raw.items()}

def _score_from_cv(cv_text: str) -> Dict[str, float]:
    if not cv_text:
        return {}
    text = _normalize_text(cv_text)
    counts: Dict[str, float] = {}
    for comp, synonyms in COMPETENCIA_SYNONYMS.items():
        freq = 0
        for term in synonyms:
            freq += text.count(_normalize_text(term))
        if freq > 0:
            counts[comp] = log(1 + freq, 2)
    if not counts:
        return {}
    max_score = max(counts.values()) or 1.0
    return {comp: round(v / max_score * 100.0, 1) for comp, v in counts.items()}

def _fuse_scores(scores_talleres: Dict[str, float], scores_cv: Dict[str, float]) -> List[Dict[str, Any]]:
    competencias = set(scores_talleres) | set(scores_cv)
    fused = []
    for comp in competencias:
        st = scores_talleres.get(comp)
        scv = scores_cv.get(comp)
        if st is not None and scv is not None:
            nivel = W_TALLERES * st + W_CV * scv
            fuente = ["talleres", "cv"]
            confianza = 0.85
        elif st is not None:
            nivel = st
            fuente = ["talleres"]
            confianza = 0.75
        else:
            nivel = scv or 0.0
            fuente = ["cv"]
            confianza = 0.65
        nivel = _safe(nivel, 0.0, 100.0)
        fused.append({"competencia": comp, "nivel": round(nivel, 1), "confianza": round(confianza, 2), "fuente": fuente})
    fused.sort(key=lambda x: -x["nivel"])
    return fused

# ====== ML ======
from app.ml.model_loader import load_model

def _build_text_for_model(cv_text: str | None, talleres: List[Dict[str, Any]] | None) -> str:
    cv = (cv_text or "").strip().lower()
    taller_tokens = ""
    if talleres:
        topics = [t.get("tema", "").strip().lower() for t in talleres if t.get("tema")]
        taller_tokens = " " + " ".join(f"topic:{t}" for t in topics)
    return (cv + taller_tokens).strip()

def _predict_with_ml(cv_text: str | None, talleres: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    pipeline, classes = load_model()  # levanta de models/pipeline_competencias.joblib
    text = _build_text_for_model(cv_text, talleres)
    proba = None
    try:
        proba = pipeline.predict_proba([text])[0]  # vector de probabilidades por clase
    except AttributeError:
        # si el estimador no tiene predict_proba (algunos modelos), usamos decision_function -> sigmoide soft
        import numpy as np
        logits = pipeline.decision_function([text])[0]
        proba = 1 / (1 + np.exp(-logits))
    # umbral simple
    threshold = float(os.getenv("ML_THRESHOLD", "0.35"))
    results = []
    for cls, p in zip(classes, proba):
        if p >= threshold:
            results.append({"competencia": cls, "nivel": round(p * 100.0, 1), "confianza": 0.9, "fuente": ["ml"]})
    # ordenar por nivel
    results.sort(key=lambda x: -x["nivel"])
    return results

def analyze_participant_profile(payload) -> Dict[str, Any]:
    use_ml = bool(getattr(payload, "useML", False))
    talleres = None
    if getattr(payload, "incluirTalleres", True) and getattr(payload, "talleres", None):
        talleres = [t.model_dump() if hasattr(t, "model_dump") else t for t in payload.talleres]
    cv_text = getattr(payload, "cvTexto", None)

    if use_ml:
        # ML directo (predice competencias)
        compet_ml = _predict_with_ml(cv_text, talleres)
        return {
            "participanteId": payload.participanteId,
            "competencias": compet_ml,
            "meta": {"mode": "ml", "threshold": os.getenv("ML_THRESHOLD", "0.35")}
        }

    # Fallback: reglas + fusión
    scores_talleres = _score_from_talleres(talleres or [])
    scores_cv = _score_from_cv(cv_text or "")
    competencias = _fuse_scores(scores_talleres, scores_cv)
    return {
        "participanteId": payload.participanteId,
        "competencias": competencias,
        "meta": {"mode": "rules", "W_TALLERES": W_TALLERES, "W_CV": W_CV}
    }
