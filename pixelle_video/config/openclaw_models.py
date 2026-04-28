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

"""Loader for the OpenClaw backend-model whitelist.

The whitelist is maintained in `openclaw_models.yaml` alongside this module so
new backend models can be added without editing Python code.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

DEFAULT_OPENCLAW_MODELS: List[Tuple[str, str]] = [
    ("qwen/qwen-max", "Qwen Max (default)"),
    ("qwen/qwen-plus", "Qwen Plus"),
    ("qwen/qwen-turbo", "Qwen Turbo"),
    ("openai/gpt-4o-mini", "OpenAI GPT-4o mini"),
    ("openai/gpt-4o", "OpenAI GPT-4o"),
    ("anthropic/claude-haiku-4-5", "Anthropic Claude Haiku 4.5"),
    ("anthropic/claude-sonnet-4-5", "Anthropic Claude Sonnet 4.5"),
    ("google/gemini-2.5-flash", "Google Gemini 2.5 Flash"),
    ("ollama/qwen2.5", "Ollama Qwen2.5 (local)"),
]

DEFAULT_OPENCLAW_MODEL_ID = "qwen/qwen-max"

_WHITELIST_FILE = Path(__file__).resolve().parent / "openclaw_models.yaml"


def load_openclaw_models() -> Tuple[str, List[Tuple[str, str]]]:
    """Load the OpenClaw backend-model whitelist.

    Returns a tuple ``(default_model_id, [(model_id, label), ...])``. Falls
    back to the in-code defaults if the YAML file cannot be read.
    """
    if not _WHITELIST_FILE.exists():
        return DEFAULT_OPENCLAW_MODEL_ID, list(DEFAULT_OPENCLAW_MODELS)
    try:
        raw = yaml.safe_load(_WHITELIST_FILE.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read {_WHITELIST_FILE}: {exc}; using defaults.")
        return DEFAULT_OPENCLAW_MODEL_ID, list(DEFAULT_OPENCLAW_MODELS)

    default_id = (raw.get("default") or DEFAULT_OPENCLAW_MODEL_ID).strip()
    entries: List[Tuple[str, str]] = []
    for item in raw.get("models") or []:
        if not isinstance(item, dict):
            continue
        model_id = (item.get("id") or "").strip()
        if not model_id:
            continue
        label = (item.get("label") or model_id).strip()
        entries.append((model_id, label))

    if not entries:
        return DEFAULT_OPENCLAW_MODEL_ID, list(DEFAULT_OPENCLAW_MODELS)
    return default_id, entries
