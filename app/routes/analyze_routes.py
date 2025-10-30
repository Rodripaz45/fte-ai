from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.services.analysis_service import analyze_participant_profile

router = APIRouter(prefix="/analyze", tags=["Analysis"])

class TallerLite(BaseModel):
    tema: str
    asistencia_pct: float = Field(ge=0, le=1)

class AnalyzeInput(BaseModel):
    participanteId: str
    talleres: list[TallerLite] | None = None
    cvTexto: str | None = None

@router.post("/profile")
def analyze_profile(payload: AnalyzeInput):
    return analyze_participant_profile(payload)
