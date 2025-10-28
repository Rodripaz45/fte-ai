from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.services.job_service import analyze_job_requirements

router = APIRouter(prefix="/analyze", tags=["Analyze / Job"])

class JobRequest(BaseModel):
    puestoTexto: str = Field(..., description="Descripción libre del puesto / necesidades")
    topK: int = Field(6, ge=1, le=20, description="Máximo de competencias a retornar")

class JobResponse(BaseModel):
    competencias: list
    meta: dict

@router.post("/job", response_model=JobResponse)
def analyze_job(req: JobRequest):
    result = analyze_job_requirements(req.puestoTexto, top_k=req.topK)
    return result
