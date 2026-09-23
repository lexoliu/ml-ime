"""Configuration read from the environment, with a repo-root ``.env`` as the source.

The LLM annotator is the one component that needs credentials. They live in a
gitignored ``.env`` at the repository root during development and come straight
from the process environment inside a Kaggle or Colab kernel, so the file is
discovered by walking upwards rather than hardcoded -- an absolute developer path
would simply be absent in a kernel.

A missing setting raises here, at construction, rather than surfacing as a 401
from the endpoint after an hour of annotation.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mlime.locate import find_upwards


class LlmSettings(BaseSettings):
    """Connection details for the OpenAI-compatible annotation endpoint."""

    model_config = SettingsConfigDict(
        env_prefix="MLIME_LLM_",
        env_file=find_upwards(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = Field(description="Root of the OpenAI-compatible API, ending in /v1")
    api_key: str = Field(description="Bearer token for the endpoint")
    model: str = Field(description="Model id to annotate with")

    @classmethod
    def load(cls) -> LlmSettings:
        """Read the settings, raising a ``ValidationError`` naming any that are unset.

        The constructor fills every field from the environment, which mypy cannot
        see, so the call is made once here rather than apologised for at each use.
        """
        return cls()  # type: ignore[call-arg]
