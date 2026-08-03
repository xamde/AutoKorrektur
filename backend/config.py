"""Application configuration settings using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUTOKORREKTUR_",
        extra="ignore",
    )

    redis_url: str | None = None
    max_daily_requests: int = 10
    enable_sdxl_load: bool = False
    sdxl_model_id: str = "runwayml/stable-diffusion-inpainting"
    strict_integrity_check: bool = False
    allowed_integrity_tokens: list[str] = ["mock-valid-token"]


settings = BackendSettings()
