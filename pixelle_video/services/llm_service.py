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
LLM (Large Language Model) Service - Direct OpenAI SDK implementation

Supports structured output via response_type parameter (Pydantic model).
"""

import json
import re
from typing import Optional, Type, TypeVar, Union

from openai import AsyncOpenAI
from pydantic import BaseModel
from loguru import logger


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Embedded-mode helpers
# ---------------------------------------------------------------------------
# yyvideoclaw launches Pixelle as a managed subprocess and injects per-process
# credentials via environment variables (see integration plan §2 / §4).
# Keeping this mapping at module scope makes it trivially mockable in tests.
_OPENCLAW_ENV_MAP: dict = {
    "provider": "PIXELLE_LLM_PROVIDER",
    "base_url": "PIXELLE_OPENCLAW_BASE_URL",
    "api_key": "PIXELLE_OPENCLAW_TOKEN",
    "agent": "PIXELLE_OPENCLAW_AGENT",
    "model": "PIXELLE_OPENCLAW_MODEL",
}

# Env flag name is re-used by api.app / health router; keep it as a single
# source of truth.
EMBEDDED_MODE_ENV = "PIXELLE_EMBEDDED_MODE"


def _is_embedded_mode(env=None) -> bool:
    """Return ``True`` iff ``PIXELLE_EMBEDDED_MODE`` is set to a truthy value.

    Recognised truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Everything else — including the variable being unset — is falsy. Passing an
    explicit mapping (instead of relying on ``os.environ``) makes unit tests
    hermetic.
    """
    import os as _os

    source = env if env is not None else _os.environ
    raw = source.get(EMBEDDED_MODE_ENV, "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class LLMService:
    """
    LLM (Large Language Model) service
    
    Direct implementation using OpenAI SDK. No capability layer needed.
    
    Supports all OpenAI SDK compatible providers:
    - OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
    - Alibaba Qwen (qwen-max, qwen-plus, qwen-turbo)
    - Anthropic Claude (claude-sonnet-4-5, claude-opus-4, claude-haiku-4)
    - DeepSeek (deepseek-chat)
    - Moonshot Kimi (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k)
    - Ollama (llama3.2, qwen2.5, mistral, codellama) - FREE & LOCAL!
    - Any custom provider with OpenAI-compatible API
    
    Usage:
        # Direct call
        answer = await pixelle_video.llm("Explain atomic habits")
        
        # With parameters
        answer = await pixelle_video.llm(
            prompt="Explain atomic habits in 3 sentences",
            temperature=0.7,
            max_tokens=2000
        )
    """
    
    def __init__(self, config: dict):
        """
        Initialize LLM service
        
        Args:
            config: Full application config dict (kept for backward compatibility)
        """
        # Note: We no longer cache config here to support hot reload
        # Config is read dynamically from config_manager in _get_config_value()
        self._client: Optional[AsyncOpenAI] = None
    
    def _get_config_value(self, key: str, default=None):
        """
        Get config value dynamically from config_manager (supports hot reload)

        In embedded mode (``PIXELLE_EMBEDDED_MODE=1``), a curated set of
        ``PIXELLE_OPENCLAW_*`` environment variables takes precedence over the
        persisted YAML config so that yyvideoclaw can inject per-subprocess
        credentials (ephemeral token, active default model, gateway URL)
        without mutating any on-disk file. This matches the zero-config
        guarantee in the integration plan (see requirements §2 / §4).

        Args:
            key: Config key name (``api_key`` / ``base_url`` / ``model`` /
                ``provider`` / ``agent``).
            default: Default value if not found.

        Returns:
            Config value.
        """
        import os

        # Env overrides are only considered when yyvideoclaw has explicitly
        # opted us into embedded mode; standalone Pixelle users keep the
        # exact legacy behaviour (config file is the sole source of truth).
        if _is_embedded_mode(os.environ):
            env_override = _OPENCLAW_ENV_MAP.get(key)
            if env_override is not None:
                raw = os.environ.get(env_override)
                if raw is not None and raw.strip():
                    return raw

        from pixelle_video.config import config_manager
        return getattr(config_manager.config.llm, key, default)

    def _resolve_request_context(self, model: Optional[str]) -> tuple[str, Optional[dict]]:
        """
        Resolve the OpenAI-SDK `model` field and any `extra_headers` the
        request should carry, based on the current llm.provider setting.

        - provider == "openclaw": route via yyvideoclaw Gateway. The OpenAI
          `model` field carries the agent target (e.g. `openclaw/llm-passthrough`)
          and the backend model is passed via the `x-openclaw-model` header so
          users can switch LLMs without restarting Pixelle. When `model` is
          unset we fall back to `qwen/qwen-max`.
        - provider == "openai" (or anything else, for legacy configs): behave
          exactly like the original direct-connect path.
        """
        provider = (self._get_config_value("provider", "openai") or "openai").strip().lower()
        backend_model = (model or self._get_config_value("model") or "").strip()
        if provider == "openclaw":
            agent = (self._get_config_value("agent") or "openclaw/llm-passthrough").strip()
            final_model = agent
            header_value = backend_model or "qwen/qwen-max"
            return final_model, {"x-openclaw-model": header_value}
        return backend_model or "gpt-3.5-turbo", None
    
    def _create_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> AsyncOpenAI:
        """
        Create OpenAI client
        
        Args:
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
        
        Returns:
            AsyncOpenAI client instance
        """
        # Get API key (priority: parameter > config)
        final_api_key = (
            api_key
            or self._get_config_value("api_key")
            or "dummy-key"  # Ollama doesn't need real key
        )
        
        # Get base URL (priority: parameter > config)
        final_base_url = (
            base_url
            or self._get_config_value("base_url")
        )
        
        # Create client
        client_kwargs = {"api_key": final_api_key}
        if final_base_url:
            client_kwargs["base_url"] = final_base_url

        # Warn when OpenClaw Gateway is configured over non-loopback plain HTTP
        if final_base_url and final_base_url.startswith("http://"):
            _is_loopback = (
                "127.0.0.1" in final_base_url
                or "localhost" in final_base_url
                or "::1" in final_base_url
            )
            if not _is_loopback:
                logger.warning(
                    f"LLM base_url uses plain http on a non-loopback host ({final_base_url}); "
                    "recommend HTTPS or SSH tunnel / tailnet for gateway calls."
                )

        return AsyncOpenAI(**client_kwargs)
    
    async def __call__(
        self,
        prompt: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_type: Optional[Type[T]] = None,
        **kwargs
    ) -> Union[str, T]:
        """
        Generate text using LLM
        
        Args:
            prompt: The prompt to generate from
            api_key: API key (optional, uses config if not provided)
            base_url: Base URL (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            max_tokens: Maximum tokens to generate
            response_type: Optional Pydantic model class for structured output.
                          If provided, returns parsed model instance instead of string.
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text (str) or parsed Pydantic model instance (if response_type provided)
        
        Examples:
            # Basic text generation
            answer = await pixelle_video.llm("Explain atomic habits")
            
            # Structured output with Pydantic model
            class MovieReview(BaseModel):
                title: str
                rating: int
                summary: str
            
            review = await pixelle_video.llm(
                prompt="Review the movie Inception",
                response_type=MovieReview
            )
            print(review.title)  # Structured access
        """
        # Create client (new instance each time to support parameter overrides)
        client = self._create_client(api_key=api_key, base_url=base_url)

        # Resolve request-level model + optional OpenClaw header override.
        # When provider=="openclaw" the OpenAI `model` carries the agent target
        # and the backend model is forwarded via x-openclaw-model header.
        final_model, extra_headers = self._resolve_request_context(model)

        logger.debug(
            f"LLM call: model={final_model}, base_url={client.base_url}, "
            f"response_type={response_type}, extra_headers={bool(extra_headers)}"
        )
        
        try:
            if response_type is not None:
                # Structured output mode - try beta.chat.completions.parse first
                return await self._call_with_structured_output(
                    client=client,
                    model=final_model,
                    prompt=prompt,
                    response_type=response_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    **kwargs
                )
            else:
                # Standard text output mode
                create_kwargs = {
                    "model": final_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if extra_headers:
                    create_kwargs["extra_headers"] = extra_headers
                response = await client.chat.completions.create(**create_kwargs)

                result = response.choices[0].message.content
                logger.debug(f"LLM response length: {len(result)} chars")

                return result

        except Exception as e:
            # Expose gateway-friendly diagnostics when OpenClaw mode is enabled.
            _diag = ""
            if extra_headers and "x-openclaw-model" in extra_headers:
                _api_key = self._get_config_value("api_key", "") or ""
                _prefix = _api_key[:4] if _api_key else "<empty>"
                _diag = (
                    f" [OpenClaw mode: gateway={client.base_url}, token_prefix={_prefix}, "
                    f"backend_model={extra_headers.get('x-openclaw-model')}; verify Gateway "
                    f"/v1/chat/completions is reachable]"
                )
            logger.error(
                f"LLM call error (model={final_model}, base_url={client.base_url}): {e}{_diag}"
            )
            raise
    
    async def _call_with_structured_output(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: str,
        response_type: Type[T],
        temperature: float,
        max_tokens: int,
        extra_headers: Optional[dict] = None,
        **kwargs
    ) -> T:
        """
        Call LLM with structured output support
        
        Uses JSON schema instruction appended to prompt for maximum compatibility
        across all OpenAI-compatible providers (Qwen, DeepSeek, etc.).
        
        Args:
            client: OpenAI client
            model: Model name
            prompt: The prompt
            response_type: Pydantic model class
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional parameters
        
        Returns:
            Parsed Pydantic model instance
        """
        # Build JSON schema instruction and append to prompt
        json_schema_instruction = self._get_json_schema_instruction(response_type)
        enhanced_prompt = f"{prompt}\n\n{json_schema_instruction}"
        
        # Call LLM with enhanced prompt
        create_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": enhanced_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if extra_headers:
            create_kwargs["extra_headers"] = extra_headers
        response = await client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content
        
        logger.debug(f"Structured output response length: {len(content)} chars")
        
        # Parse JSON from response content
        return self._parse_response_as_model(content, response_type)
    
    def _get_json_schema_instruction(self, response_type: Type[T]) -> str:
        """
        Generate JSON schema instruction for LLM fallback mode
        
        Args:
            response_type: Pydantic model class
        
        Returns:
            Formatted instruction string with JSON schema
        """
        try:
            # Get JSON schema from Pydantic model
            schema = response_type.model_json_schema()
            schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
            
            return f"""## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text).
The JSON must strictly follow this schema:

```json
{schema_str}
```

Output ONLY the JSON object, nothing else."""
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema: {e}")
            return """## IMPORTANT: JSON Output Format Required
You MUST respond with ONLY a valid JSON object (no markdown, no extra text)."""
    
    def _parse_response_as_model(self, content: str, response_type: Type[T]) -> T:
        """
        Parse LLM response content as Pydantic model
        
        Args:
            content: Raw LLM response text
            response_type: Target Pydantic model class
        
        Returns:
            Parsed model instance
        """
        # Try direct JSON parsing first
        try:
            data = json.loads(content)
            return response_type.model_validate(data)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_pattern = r'```(?:json)?\s*([\s\S]+?)\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return response_type.model_validate(data)
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the text
        brace_start = content.find('{')
        brace_end = content.rfind('}')
        if brace_start != -1 and brace_end > brace_start:
            try:
                json_str = content[brace_start:brace_end + 1]
                data = json.loads(json_str)
                return response_type.model_validate(data)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to parse LLM response as {response_type.__name__}: {content[:200]}...")
    
    @property
    def active(self) -> str:
        """
        Get active model name
        
        Returns:
            Active model name
        
        Example:
            print(f"Using model: {pixelle_video.llm.active}")
        """
        return self._get_config_value("model", "gpt-3.5-turbo")
    
    def __repr__(self) -> str:
        """String representation"""
        model = self.active
        base_url = self._get_config_value("base_url", "default")
        return f"<LLMService model={model!r} base_url={base_url!r}>"

