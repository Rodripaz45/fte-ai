import os
import joblib
from functools import lru_cache

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline_competencias.joblib")

@lru_cache(maxsize=1)
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    thresholds = dict(artifact.get("thresholds", {}))
    return pipeline, list(classes), thresholds
