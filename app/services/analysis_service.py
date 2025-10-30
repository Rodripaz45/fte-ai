from __future__ import annotations
from collections import defaultdict
from math import log
from typing import Any, Dict, List, Tuple
import os

# ====== Reglas de fallback alineadas con el dataset ======
# Nota: estos mapas solo se usan cuando el modelo ML no está disponible.
# Etiquetas y nombres iguales a los presentes en el dataset/entrenamiento.
TOPIC_TO_COMPETENCIAS: Dict[str, List[Tuple[str, float]]] = {
    # Ofimática / Análisis de datos
    "excel": [("Analisis de Datos", 0.8), ("Ofimática", 1.0)],
    "power bi": [("Analisis de Datos", 1.0), ("Ofimática", 0.8)],
    "sql": [("Analisis de Datos", 1.0)],
    "python": [("Analisis de Datos", 0.9), ("Ingeniería de Software", 0.8)],
    "dax": [("Analisis de Datos", 1.0)],
    "etl": [("Analisis de Datos", 1.0)],
    # Software/DevOps/SRE/Ciberseguridad/Datos
    "react": [("Ingeniería de Software", 1.0)],
    "node": [("Ingeniería de Software", 1.0)],
    "apis": [("Ingeniería de Software", 0.9)],
    "docker": [("DevOps/SRE", 1.0)],
    "kubernetes": [("DevOps/SRE", 1.0)],
    "terraform": [("DevOps/SRE", 1.0)],
    "ci/cd": [("DevOps/SRE", 1.0)],
    "sre": [("DevOps/SRE", 1.0)],
    "owasp": [("Ciberseguridad", 1.0)],
    "siem": [("Ciberseguridad", 1.0)],
    "airflow": [("Ingeniería de Datos", 1.0)],
    "bigquery": [("Ingeniería de Datos", 1.0)],
    # Operaciones/Industria/Calidad/Seguridad
    "iso 9001": [("Calidad", 1.0)],
    "hse": [("Seguridad e Higiene", 1.0)],
    "cmms": [("Mantenimiento", 1.0)],
    "lean": [("Producción", 1.0)],
    "supply": [("Logística", 1.0)],
    # Gestión/Negocio
    "ms project": [("Gestion de Proyectos", 1.0)],
    "presupuesto": [("Finanzas", 1.0)],
    "crm": [("Ventas", 1.0), ("Atención al Cliente", 0.8)],
    # Sectores adicionales
    "fotovoltaica": [("Energías Renovables", 1.0)],
    "pozos": [("Petróleo y Gas", 1.0)],
    "voladura": [("Minería", 1.0)],
    "gis": [("Recursos Hídricos", 1.0)],
    "haccp": [("Agroindustria", 1.0), ("Calidad", 0.8)],
    "actuarial": [("Seguros", 1.0)],
    "lte": [("Telecomunicaciones", 1.0)],
}

COMPETENCIA_SYNONYMS: Dict[str, List[str]] = {
    "Analisis de Datos": ["excel", "sql", "power bi", "tableau", "dashboards", "etl", "analytics"],
    "Ofimática": ["excel", "office", "powerpoint", "word", "power query"],
    "Ingeniería de Software": ["python", "node", "javascript", "apis", "testing", "desarrollo"],
    "DevOps/SRE": ["docker", "kubernetes", "terraform", "observabilidad", "slo", "sli", "cicd", "ci/cd"],
    "Ciberseguridad": ["owasp", "siem", "hardening", "cis", "parches", "pentest"],
    "Ingeniería de Datos": ["airflow", "spark", "bigquery", "etl", "dataops"],
    "Calidad": ["iso 9001", "auditorias", "no conformidades", "spc", "control de calidad"],
    "Seguridad e Higiene": ["hse", "epi", "bioseguridad", "simulacros"],
    "Mantenimiento": ["cmms", "preventivo", "correctivo", "rca"],
    "Producción": ["lean", "smed", "oee", "kaizen"],
    "Logística": ["wms", "ruteo", "inventario", "supply"],
    "Finanzas": ["presupuesto", "flujo de caja", "variaciones", "control de gestión"],
    "Gestion de Proyectos": ["ms project", "kanban", "scrum", "raci", "cronograma"],
    "Ventas": ["crm", "prospección", "negociación", "pipeline"],
    "Atención al Cliente": ["nps", "postventa", "reclamos", "contact center"],
    "Energías Renovables": ["fotovoltaica", "pvsyst", "inversores"],
    "Petróleo y Gas": ["pozos", "workover", "scada"],
    "Minería": ["voladura", "mina", "cielo abierto"],
    "Recursos Hídricos": ["hidrologia", "gis", "permisos"],
    "Agroindustria": ["haccp", "bpm", "trazabilidad"],
    "Seguros": ["siniestros", "pricing", "actuarial"],
    "Telecomunicaciones": ["lte", "5g", "kpis"],
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
    pipeline, classes, metadata = load_model()  # levanta de models/pipeline_competencias.joblib
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
    threshold = float(os.getenv("ML_THRESHOLD", metadata.get("best_threshold", metadata.get("threshold", "0.35"))))

    # resultados por encima del umbral
    above = [
        {"competencia": cls, "nivel": round(float(p) * 100.0, 1), "confianza": 0.9, "fuente": ["ml"]}
        for cls, p in zip(classes, proba)
        if float(p) >= threshold
    ]
    above.sort(key=lambda x: -x["nivel"])

    # Fallback: si hay muy pocos por encima del umbral, completar con los mejores siguientes
    min_results = int(os.getenv("ML_MIN_RESULTS", "3"))
    min_prob_floor = float(os.getenv("ML_MIN_PROB_FLOOR", "0.25"))
    if len(above) >= min_results:
        return above

    # Tomar candidatos restantes por probabilidad
    pairs = list(zip(classes, proba))
    # excluir ya elegidos
    chosen = {c["competencia"] for c in above}
    remaining = [(c, float(p)) for c, p in pairs if c not in chosen]
    remaining.sort(key=lambda x: -x[1])
    for cls, p in remaining:
        if p < min_prob_floor:
            break
        above.append({"competencia": cls, "nivel": round(p * 100.0, 1), "confianza": 0.75, "fuente": ["ml"]})
        if len(above) >= min_results:
            break
    return above

def analyze_participant_profile(payload) -> Dict[str, Any]:
    talleres = None
    if getattr(payload, "talleres", None):
        talleres = [t.model_dump() if hasattr(t, "model_dump") else t for t in payload.talleres]
    cv_text = getattr(payload, "cvTexto", None)

    # Siempre usar ML si el modelo está disponible; si no, caer a reglas
    compet_ml = _predict_with_ml(cv_text, talleres)
    if compet_ml:
        return {
            "participanteId": payload.participanteId,
            "competencias": compet_ml,
            "meta": {"mode": "ml"}
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
