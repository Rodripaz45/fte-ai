import jwt
from fastapi import Security, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.config import settings

http_bearer = HTTPBearer(auto_error=False)

def verify_service_bearer(credentials: HTTPAuthorizationCredentials = Security(http_bearer)):
    if not credentials or not credentials.scheme.lower() == "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = credentials.credentials
    try:
        jwt.decode(token, settings.SERVICE_JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid service token")
