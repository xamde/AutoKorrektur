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
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB default
    enable_sdxl_load: bool = True
    sdxl_model_id: str = "runwayml/stable-diffusion-inpainting"
    strict_integrity_check: bool = True
    allowed_integrity_tokens: list[str] = ["mock-valid-token"]
    google_application_credentials: str | None = None
    android_package_name: str = "de.konradvoelkel.android.autokorrektur"


settings = BackendSettings()
