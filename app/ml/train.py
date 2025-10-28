from __future__ import annotations
import os
import argparse
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# --- configuración por defecto ---
DATA_PATH = os.getenv("DATA_PATH", "data/dataset_competencias.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline_competencias.joblib")

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normaliza columnas
    for col in ["cv_texto", "talleres", "competencias"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")
    df["cv_texto"] = df["cv_texto"].fillna("").astype(str)
    df["talleres"] = df["talleres"].fillna("").astype(str)
    df["competencias"] = df["competencias"].fillna("").astype(str)
    return df

def build_text(row) -> str:
    """
    Concatenamos CV + talleres (como tokens) para mejorar señal.
    Ej: '...texto del cv...' + ' topic:python topic:sql '
    """
    cv = row["cv_texto"].strip().lower()
    talleres = [t.strip().lower() for t in row["talleres"].split(",") if t.strip()]
    taller_tokens = " ".join([f"topic:{t}" for t in talleres])
    return f"{cv} {taller_tokens}"

def main(data_path: str, model_path: str, test_size: float = 0.2, random_state: int = 42):
    print(f"[train] leyendo dataset: {data_path}")
    df = load_dataset(data_path)

    # Texto de entrada = CV + talleres
    X_text = df.apply(build_text, axis=1).tolist()

    # Etiquetas multilabel
    y_list = [ [c.strip() for c in comps.split(",") if c.strip()] for comps in df["competencias"].tolist() ]
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_text, Y, test_size=test_size, random_state=random_state)

    # Pipeline TF-IDF + OneVsRest(LogReg)
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            OneVsRestClassifier(
                LogisticRegression(max_iter=400, class_weight="balanced", solver="liblinear")
            ),
        ),
    ])

    print("[train] entrenando modelo...")
    pipeline.fit(X_train, y_train)

    print("[train] evaluación...")
    target_names = mlb.classes_
    y_pred_proba = pipeline.predict_proba(X_test)
    thresholds, y_pred = _compute_thresholds(y_test, y_pred_proba, target_names)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Guardar pipeline + clases (etiquetas)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    artifact = {"pipeline": pipeline, "classes": target_names, "thresholds": thresholds}
    joblib.dump(artifact, model_path)
    print(f"[train] modelo guardado en: {model_path}")
    print("[train] umbrales por clase:")
    for cls, threshold in thresholds.items():
        print(f"  - {cls}: {threshold:.3f}")
    print("[train] listo ✅")


def _compute_thresholds(
    y_true: np.ndarray, y_scores: np.ndarray, classes: List[str], default_threshold: float = 0.35
) -> tuple[Dict[str, float], np.ndarray]:
    """Calcula un umbral óptimo por clase maximizando F1 en el conjunto de prueba."""

    thresholds: Dict[str, float] = {}
    y_pred = np.zeros_like(y_true)

    for idx, cls in enumerate(classes):
        true_col = y_true[:, idx]
        score_col = y_scores[:, idx]

        if true_col.sum() == 0:
            # Sin ejemplos positivos → se mantiene umbral por defecto
            threshold = default_threshold
        else:
            precision, recall, thresh = precision_recall_curve(true_col, score_col)
            if thresh.size == 0:
                threshold = default_threshold
            else:
                # precision/recall tienen un elemento adicional (para threshold=0)
                f1_scores = (2 * precision[:-1] * recall[:-1]) / (
                    precision[:-1] + recall[:-1] + 1e-12
                )
                best_idx = int(np.nanargmax(f1_scores))
                threshold = float(thresh[best_idx])

        thresholds[cls] = threshold
        y_pred[:, idx] = (score_col >= threshold).astype(int)

    return thresholds, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--out", default=MODEL_PATH)
    args = parser.parse_args()
    main(args.data, args.out)
