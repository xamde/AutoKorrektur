from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUTOKORREKTUR_",
        extra="ignore",
    )

    redis_url: str | None = Field(
        default=None,
        description="Optional Redis connection URL (e.g., redis://localhost:6379/0) for distributed rate limiting",
    )
    max_daily_requests: int = Field(
        default=10,
        description="Maximum number of free cloud inpainting requests allowed per device/IP per day",
    )
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum combined upload payload size in bytes (default 10 MB)",
    )
    enable_sdxl_load: bool = Field(
        default=True,
        description="Whether to load the neural inpainting pipeline into GPU/CPU RAM at startup",
    )
    sdxl_model_id: str = Field(
        default="runwayml/stable-diffusion-inpainting",
        description="HuggingFace repository ID or local path for the diffusion inpainting model (defaults to lightweight SD 1.5 inpainting fallback; can be configured to diffusers/stable-diffusion-xl-1.0-inpainting-0.1)",
    )
    strict_integrity_check: bool = Field(
        default=True,
        description="Whether to enforce strict Google Play Integrity attestation token verification",
    )
    allowed_integrity_tokens: list[str] = Field(
        default_factory=list,
        description="List of mock Play Integrity tokens permitted in development/test environments",
    )
    google_application_credentials: str | None = Field(
        default=None,
        description="Optional path to Google Cloud service account JSON for Play Integrity API verification",
    )
    android_package_name: str = Field(
        default="de.konradvoelkel.android.autokorrektur",
        description="Expected Android package name validated against Play Integrity claims",
    )


import functools


@functools.lru_cache
def get_settings() -> BackendSettings:
    """Returns a cached instance of BackendSettings."""
    return BackendSettings()


settings = get_settings()
