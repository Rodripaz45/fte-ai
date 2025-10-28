import os
import joblib
from functools import lru_cache

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline_competencias.joblib")

@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]   
    classes = artifact.get("classes")
    if classes is None and "mlb" in artifact:
        classes = getattr(artifact["mlb"], "classes_", None)
    if classes is None:
        classes = []

    return pipeline, list(classes)
