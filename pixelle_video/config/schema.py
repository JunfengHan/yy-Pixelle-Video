# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration schema with Pydantic models

Single source of truth for all configuration defaults and validation.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# Canonical set of provider modes understood by :class:`LLMConfig`.
# Kept as a module-level tuple so the same source of truth is used both by the
# Pydantic validator and by callers (llm_service, verify script) that want to
# introspect/log the active mode.
LLM_PROVIDER_MODES = ("openai", "openclaw")


class LLMConfig(BaseModel):
    """LLM configuration"""
    api_key: str = Field(default="", description="LLM API Key")
    base_url: str = Field(default="", description="LLM API Base URL")
    model: str = Field(default="", description="LLM Model Name")
    # Optional integration with an OpenClaw / yyvideoclaw Gateway.
    # When provider == "openclaw":
    #   - base_url points to the Gateway's OpenAI-compat endpoint (e.g. http://127.0.0.1:18789/v1)
    #   - api_key is the Gateway bearer token
    #   - `agent` is sent as the OpenAI `model` field (agent target)
    #   - `model` is forwarded as the `x-openclaw-model` HTTP header to override
    #     the agent's underlying LLM at request time
    # When provider == "openai" (default for legacy configs missing this field),
    # the service behaves exactly as before (direct OpenAI-compatible call).
    provider: str = Field(
        default="openai",
        description=(
            'LLM provider mode: "openai" (direct OpenAI-compatible) or "openclaw" '
            "(via yyvideoclaw Gateway). Unknown values fall back to \"openai\" so "
            "older configs keep working after upgrade."
        ),
    )
    agent: str = Field(
        default="openclaw/llm-passthrough",
        description='OpenClaw agent target, used as the OpenAI `model` field when provider=="openclaw".',
    )

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> str:
        """
        Normalise and soft-validate the provider field.

        Behaviour:
          - ``None`` / empty / whitespace → ``"openai"`` (back-compat default).
          - Case/whitespace insensitive match against :data:`LLM_PROVIDER_MODES`.
          - Any other value → ``"openai"`` (intentionally forgiving so a typo
            in a user-managed YAML config doesn't brick the service; the
            loader surfaces a warning elsewhere).
        """
        if value is None:
            return "openai"
        if not isinstance(value, str):
            return "openai"
        normalised = value.strip().lower()
        if not normalised:
            return "openai"
        return normalised if normalised in LLM_PROVIDER_MODES else "openai"


class TTSLocalConfig(BaseModel):
    """Local TTS configuration (Edge TTS)"""
    voice: str = Field(default="zh-CN-YunjianNeural", description="Edge TTS voice ID")
    speed: float = Field(default=1.2, ge=0.5, le=2.0, description="Speech speed multiplier (0.5-2.0)")


class TTSComfyUIConfig(BaseModel):
    """ComfyUI TTS configuration"""
    default_workflow: Optional[str] = Field(default=None, description="Default TTS workflow (optional)")


class TTSSubConfig(BaseModel):
    """TTS-specific configuration (under comfyui.tts)"""
    inference_mode: str = Field(default="local", description="TTS inference mode: 'local' or 'comfyui'")
    local: TTSLocalConfig = Field(default_factory=TTSLocalConfig, description="Local TTS (Edge TTS) configuration")
    comfyui: TTSComfyUIConfig = Field(default_factory=TTSComfyUIConfig, description="ComfyUI TTS configuration")
    
    # Backward compatibility: keep default_workflow at top level
    @property
    def default_workflow(self) -> Optional[str]:
        """Get default workflow (for backward compatibility)"""
        return self.comfyui.default_workflow


class ImageSubConfig(BaseModel):
    """Image-specific configuration (under comfyui.image)"""
    default_workflow: Optional[str] = Field(default=None, description="Default image workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all image generation"
    )


class VideoSubConfig(BaseModel):
    """Video-specific configuration (under comfyui.video)"""
    default_workflow: Optional[str] = Field(default=None, description="Default video workflow (optional)")
    prompt_prefix: str = Field(
        default="Minimalist black-and-white matchstick figure style illustration, clean lines, simple sketch style",
        description="Prompt prefix for all video generation"
    )


class ComfyUIConfig(BaseModel):
    """ComfyUI configuration (includes global settings and service-specific configs)"""
    comfyui_url: str = Field(default="http://127.0.0.1:8188", description="ComfyUI Server URL")
    comfyui_api_key: Optional[str] = Field(default=None, description="ComfyUI API Key (optional)")
    runninghub_api_key: Optional[str] = Field(default=None, description="RunningHub API Key (optional)")
    runninghub_concurrent_limit: int = Field(default=1, ge=1, le=10, description="RunningHub concurrent execution limit (1-10)")
    runninghub_instance_type: Optional[str] = Field(default=None, description="RunningHub instance type (optional, set to 'plus' for 48GB VRAM)")
    tts: TTSSubConfig = Field(default_factory=TTSSubConfig, description="TTS-specific configuration")
    image: ImageSubConfig = Field(default_factory=ImageSubConfig, description="Image-specific configuration")
    video: VideoSubConfig = Field(default_factory=VideoSubConfig, description="Video-specific configuration")


class TemplateConfig(BaseModel):
    """Template configuration"""
    default_template: str = Field(
        default="1080x1920/default.html",
        description="Default frame template path"
    )


class PixelleVideoConfig(BaseModel):
    """Pixelle-Video main configuration"""
    project_name: str = Field(default="Pixelle-Video", description="Project name")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    comfyui: ComfyUIConfig = Field(default_factory=ComfyUIConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    
    def is_llm_configured(self) -> bool:
        """Check if LLM is properly configured"""
        if self.llm.provider == "openclaw":
            # For OpenClaw gateway mode, `model` may be overridden per-request
            # via the x-openclaw-model header; api_key+base_url+agent are the
            # minimum required fields.
            return bool(
                self.llm.api_key and self.llm.api_key.strip() and
                self.llm.base_url and self.llm.base_url.strip() and
                self.llm.agent and self.llm.agent.strip()
            )
        return bool(
            self.llm.api_key and self.llm.api_key.strip() and
            self.llm.base_url and self.llm.base_url.strip() and
            self.llm.model and self.llm.model.strip()
        )
    
    def validate_required(self) -> bool:
        """Validate required configuration"""
        return self.is_llm_configured()
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)"""
        return self.model_dump()

