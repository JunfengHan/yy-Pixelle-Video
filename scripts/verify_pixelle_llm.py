#!/usr/bin/env python3
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

"""Standalone verifier: `Pixelle env → yyvideoclaw Gateway → backend LLM`.

Independent of the Pixelle main pipeline; sends one minimal chat request via
the OpenAI-compatible Gateway endpoint and prints the outcome.

Usage:
    # Reads config from environment variables or config.yaml defaults.
    python scripts/verify_pixelle_llm.py

    # Or pass everything explicitly:
    python scripts/verify_pixelle_llm.py \\
        --base-url http://127.0.0.1:18789/v1 \\
        --api-key "$OPENCLAW_GATEWAY_TOKEN" \\
        --agent openclaw/llm-passthrough \\
        --model qwen/qwen-max
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx


def _load_from_config_yaml() -> dict:
    """Best-effort load of config.yaml's llm section (for default values)."""
    try:
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            return {}
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data.get("llm") or {}
    except Exception:
        return {}


def _run_embedded_handshake() -> int:
    """
    Machine-readable probe used by yyvideoclaw's Pixelle supervisor.

    Reads the curated ``PIXELLE_OPENCLAW_*`` env block that yyvideoclaw
    injects, then fires a single ``/chat/completions`` call to make sure the
    Gateway is reachable with the ephemeral token before the video pipeline
    actually starts producing work. Output is always one line of JSON on
    stdout, regardless of success or failure, with a non-zero exit code on
    any error — this keeps parsing trivial on the Node-side supervisor.
    """
    base_url = (os.environ.get("PIXELLE_OPENCLAW_BASE_URL") or "").strip()
    token = (os.environ.get("PIXELLE_OPENCLAW_TOKEN") or "").strip()
    agent = (os.environ.get("PIXELLE_OPENCLAW_AGENT") or "openclaw/llm-passthrough").strip()
    model = (os.environ.get("PIXELLE_OPENCLAW_MODEL") or "qwen/qwen-max").strip()

    def _emit(payload: dict) -> None:
        # Newline-terminated JSON keeps partial reads on the supervisor side
        # unambiguous (one event per line).
        print(json.dumps(payload, ensure_ascii=False))

    if not base_url or not token:
        _emit({
            "ok": False,
            "stage": "env",
            "error": "PIXELLE_OPENCLAW_BASE_URL and PIXELLE_OPENCLAW_TOKEN must both be set in embedded mode",
        })
        return 2

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-openclaw-model": model,
    }
    body = {
        "model": agent,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=10.0)
    except httpx.HTTPError as exc:
        _emit({"ok": False, "stage": "network", "error": str(exc), "base_url": base_url})
        return 1

    if resp.status_code != 200:
        _emit({
            "ok": False,
            "stage": "http",
            "status": resp.status_code,
            # Truncate response body to avoid leaking large payloads into logs.
            "body_snippet": resp.text[:200],
        })
        return 1

    try:
        payload = resp.json()
        choice = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "stage": "parse", "error": str(exc)})
        return 1

    _emit({
        "ok": True,
        "base_url": base_url,
        "agent": agent,
        "model": model,
        # Never echo the full reply — it may be arbitrarily long; the caller
        # only needs to know the handshake succeeded.
        "reply_length": len(choice or ""),
    })
    return 0


def main() -> int:
    # Fast path: machine-readable handshake mode used by the yyvideoclaw
    # supervisor during Pixelle subprocess startup (see integration plan §6.4).
    # We detect this before the main argparse runs because embedded callers
    # should never need to pass any other flag.
    if "--embedded-handshake" in sys.argv[1:]:
        return _run_embedded_handshake()

    cfg_llm = _load_from_config_yaml()
    parser = argparse.ArgumentParser(description="Verify Pixelle ↔ yyvideoclaw Gateway link.")
    parser.add_argument(
        "--embedded-handshake",
        action="store_true",
        help="Machine-readable probe for yyvideoclaw supervisor (reads PIXELLE_OPENCLAW_* env).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENCLAW_GATEWAY_URL") or cfg_llm.get("base_url") or "http://127.0.0.1:18789/v1",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENCLAW_GATEWAY_TOKEN") or cfg_llm.get("api_key") or "",
    )
    parser.add_argument(
        "--agent",
        default=cfg_llm.get("agent") or "openclaw/llm-passthrough",
        help="OpenClaw agent target (sent as the OpenAI 'model' field).",
    )
    parser.add_argument(
        "--model",
        default=cfg_llm.get("model") or "qwen/qwen-max",
        help="Backend model id (sent via x-openclaw-model header).",
    )
    parser.add_argument("--prompt", default="ping")
    parser.add_argument("--max-tokens", type=int, default=5)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: missing gateway token. Set --api-key or $OPENCLAW_GATEWAY_TOKEN.", file=sys.stderr)
        return 2

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
        "x-openclaw-model": args.model,
    }
    body = {
        "model": args.agent,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }

    token_preview = args.api_key[:4] + "…" if len(args.api_key) > 4 else "<short>"
    print(
        f"[verify_pixelle_llm] POST {url}\n"
        f"                     agent  = {args.agent}\n"
        f"                     model  = {args.model}\n"
        f"                     token  = {token_preview}"
    )

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=20.0)
    except httpx.HTTPError as exc:
        print(f"❌ Network error: {exc}", file=sys.stderr)
        return 1

    if resp.status_code != 200:
        snippet = resp.text[:400].replace("\n", " ")
        print(f"❌ HTTP {resp.status_code}: {snippet}", file=sys.stderr)
        return 1

    try:
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Unexpected response shape: {exc}\n{resp.text[:400]}", file=sys.stderr)
        return 1

    print("✅ Link OK.")
    print(f"   Reply (truncated): {content[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
