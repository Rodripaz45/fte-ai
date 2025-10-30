from __future__ import annotations
import os
import argparse
import unicodedata
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer
import joblib
import numpy as np

# --- configuración por defecto ---
DATA_PATH = os.getenv("DATA_PATH", "data/dataset_competencias.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline_competencias.joblib")

SPANISH_STOP_WORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "porque", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto",
    "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo",
    "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras",
    "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos",
    "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
    "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos",
    "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría",
    "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais",
    "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera",
    "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses", "estuviésemos",
    "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad", "he",
    "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayáis", "hayan", "habré",
    "habrás", "habrá", "habremos", "habréis", "habrán", "habría", "habrías", "habríamos", "habríais",
    "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube", "hubiste", "hubo", "hubimos",
    "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran", "hubiese",
    "hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos",
    "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean",
    "seré", "serás", "será", "seremos", "seréis", "serán", "sería", "serías", "seríamos", "seríais",
    "serían", "era", "eras", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis",
    "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran", "fuese", "fueses", "fuésemos",
    "fueseis", "fuesen", "siendo", "sido", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen",
    "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá", "tendremos",
    "tendréis", "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías",
    "teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron",
    "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos",
    "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida", "tenidos", "tenidas", "tened",
]


def _normalize_stopwords(words: list[str]) -> list[str]:
    normalized = set()
    for w in words:
        normalized.add(w)
        normalized.add(
            unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode("ascii")
        )
    normalized.discard("")
    return sorted(normalized)

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

def _build_pipeline() -> Pipeline:
    """Crea el pipeline base con TF-IDF y OneVsRest(LogisticRegression)."""
    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    stop_words = _normalize_stopwords(SPANISH_STOP_WORDS)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            strip_accents="unicode",
            sublinear_tf=True,
            stop_words=stop_words,
        )),
        ("clf", OneVsRestClassifier(logistic, n_jobs=-1)),
    ])


def _run_hyperparameter_search(pipeline: Pipeline, X_train, y_train):
    """
    Realiza una búsqueda de hiperparámetros sencilla para refinar el modelo.

    Esto ayuda a que el modelo resultante generalice mejor, sobre todo cuando
    se agreguen nuevos datos al dataset.
    """
    scorer = make_scorer(f1_score, average="macro")
    param_grid = {
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__min_df": [1, 2],
        "clf__estimator__C": [0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=os.cpu_count() or 1,
        verbose=2,
    )

    print("[train] buscando mejores hiperparámetros (GridSearchCV)...")
    search.fit(X_train, y_train)
    print(f"[train] mejores hiperparámetros: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def _select_best_threshold(pipeline: Pipeline, X_valid, y_valid, thresholds=None, beta: float = 2.0, max_threshold: float = 0.40):
    if thresholds is None:
        # Rango más amplio y granular para evitar ser demasiado estricto
        thresholds = [round(t, 2) for t in np.linspace(0.15, 0.6, 46)]
    # Obtener probabilidades/logits
    try:
        proba = pipeline.predict_proba(X_valid)
    except Exception:
        logits = pipeline.decision_function(X_valid)
        proba = 1 / (1 + np.exp(-logits))
    best_t = 0.30
    best_f2 = -1.0
    for t in thresholds:
        # Limitar búsqueda para evitar umbrales excesivos en datasets pequeños
        if t > max_threshold:
            continue
        y_pred = (proba >= t).astype(int)
        f2 = fbeta_score(y_valid, y_pred, beta=beta, average="macro", zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_t = float(t)
    return best_t, best_f2


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

    pipeline = _build_pipeline()
    pipeline, best_params = _run_hyperparameter_search(pipeline, X_train, y_train)

    print("[train] seleccionando umbral óptimo (macro-F2, favorece recall)...")
    best_threshold, best_f2 = _select_best_threshold(pipeline, X_test, y_test)

    print("[train] evaluación...")
    # Predicción binaria usando umbral óptimo
    try:
        proba_test = pipeline.predict_proba(X_test)
    except Exception:
        logits = pipeline.decision_function(X_test)
        proba_test = 1 / (1 + np.exp(-logits))
    y_pred = (proba_test >= best_threshold).astype(int)
    target_names = mlb.classes_
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Guardar pipeline + clases (etiquetas)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Cardinalidad de etiquetas (promedio de etiquetas por muestra) para referencia en servicio
    label_cardinality = float(np.mean(y_train.sum(axis=1)))

    artifact = {
        "pipeline": pipeline,
        "classes": target_names,
        "mlb": mlb,
        "metadata": {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "cv_macro_f2": float(best_f2),
            "label_cardinality": label_cardinality,
        },
    }
    joblib.dump(artifact, model_path)
    print(f"[train] modelo guardado en: {model_path}")
    print("[train] listo OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--out", default=MODEL_PATH)
    args = parser.parse_args()
    main(args.data, args.out)
