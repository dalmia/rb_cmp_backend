import os
from pydantic_settings import BaseSettings
from phoenix.otel import register


class Settings(BaseSettings):
    openai_api_key: str
    phoenix_api_key: str
    phoenix_endpoint: str
    frappe_backend_base_url: str
    frappe_backend_client_id: str
    frappe_backend_client_secret: str
    database_url: str
    env: str

    class Config:
        env_file = ".env"


settings = Settings()


os.environ["PHOENIX_API_KEY"] = settings.phoenix_api_key
tracer_provider = register(
    protocol="http/protobuf",
    project_name=f"cmp-{settings.env or 'development'}",
    auto_instrument=True,
    batch=True,
    endpoint=settings.phoenix_endpoint,
)
tracer = tracer_provider.get_tracer(__name__)
