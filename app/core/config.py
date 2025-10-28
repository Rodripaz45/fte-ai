from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FTE_API_URL: str = "http://localhost:4000"
    SERVICE_JWT_SECRET: str = "super_secret_key"

    model_config = {"env_file": ".env"}


settings = Settings()
