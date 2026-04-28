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
Pixelle-Video FastAPI Application

Main FastAPI app with all routers and middleware.

Run this script to start the FastAPI server:
    uv run python api/app.py
    
Or with custom settings:
    uv run python api/app.py --host 0.0.0.0 --port 8080 --reload
"""

import sys
from pathlib import Path

# Add project root to sys.path for module imports
# This ensures imports work correctly in both development and packaged environments
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import api_config
from api.tasks import task_manager
from api.dependencies import shutdown_pixelle_video
from pixelle_video.services.llm_service import EMBEDDED_MODE_ENV, _is_embedded_mode

# Import routers
from api.routers import (
    health_router,
    llm_router,
    tts_router,
    image_router,
    content_router,
    video_router,
    tasks_router,
    files_router,
    resources_router,
    frame_router,
)


# ---------------------------------------------------------------------------
# Embedded-mode plumbing
# ---------------------------------------------------------------------------
# When Pixelle is launched by yyvideoclaw, we want:
#   1. A clear "[embedded]" prefix on every log line for cross-process triage.
#   2. Loopback-only binding (defence-in-depth; requirement §6.5, §8.1).
#   3. Standalone mode completely untouched so upstream users keep shipping.
#
# All three live here (not in `__main__`) so uvicorn launches started via
# e.g. ``uvicorn api.app:app`` also pick them up.
EMBEDDED_MODE = _is_embedded_mode()


def _install_embedded_logging() -> None:
    """Prefix every log record with "[embedded]" when running under yyvideoclaw.

    Runs exactly once at import time; safe to no-op if loguru has already been
    customised by the host (the patch is idempotent).
    """
    # ``extra`` is the recommended, non-destructive way to augment loguru.
    # We replace the default sink with one that always carries the embedded
    # tag, preserving the original formatting otherwise.
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "[embedded] <green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        enqueue=True,  # safe for multi-process / subprocess-redirected stderr
    )


if EMBEDDED_MODE:
    _install_embedded_logging()
    logger.info("Pixelle starting under yyvideoclaw embedded mode.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Pixelle-Video API...")
    await task_manager.start()
    logger.info("✅ Pixelle-Video API started successfully\n")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Pixelle-Video API...")
    await task_manager.stop()
    await shutdown_pixelle_video()
    logger.info("✅ Pixelle-Video API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Pixelle-Video API",
    description="""
    ## Pixelle-Video - AI Video Generation Platform API
    
    ### Features
    - 🤖 **LLM**: Large language model integration
    - 🔊 **TTS**: Text-to-speech synthesis
    - 🎨 **Image**: AI image generation
    - 📝 **Content**: Automated content generation
    - 🎬 **Video**: End-to-end video generation
    
    ### Video Generation Modes
    - **Sync**: `/api/video/generate/sync` - For small videos (< 30s)
    - **Async**: `/api/video/generate/async` - For large videos with task tracking
    
    ### Getting Started
    1. Check health: `GET /health`
    2. Generate narrations: `POST /api/content/narration`
    3. Generate video: `POST /api/video/generate/sync` or `/async`
    4. Track task progress: `GET /api/tasks/{task_id}`
    """,
    version="0.1.0",
    docs_url=api_config.docs_url,
    redoc_url=api_config.redoc_url,
    openapi_url=api_config.openapi_url,
    lifespan=lifespan,
)

# Add CORS middleware
if api_config.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS enabled for origins: {api_config.cors_origins}")

# Include routers
# Health check (no prefix)
app.include_router(health_router)

# API routers (with /api prefix)
app.include_router(llm_router, prefix=api_config.api_prefix)
app.include_router(tts_router, prefix=api_config.api_prefix)
app.include_router(image_router, prefix=api_config.api_prefix)
app.include_router(content_router, prefix=api_config.api_prefix)
app.include_router(video_router, prefix=api_config.api_prefix)
app.include_router(tasks_router, prefix=api_config.api_prefix)
app.include_router(files_router, prefix=api_config.api_prefix)
app.include_router(resources_router, prefix=api_config.api_prefix)
app.include_router(frame_router, prefix=api_config.api_prefix)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Pixelle-Video API",
        "version": "0.1.0",
        "docs": api_config.docs_url,
        "health": "/health",
        "api": {
            "llm": f"{api_config.api_prefix}/llm",
            "tts": f"{api_config.api_prefix}/tts",
            "image": f"{api_config.api_prefix}/image",
            "content": f"{api_config.api_prefix}/content",
            "video": f"{api_config.api_prefix}/video",
            "tasks": f"{api_config.api_prefix}/tasks",
            "files": f"{api_config.api_prefix}/files",
            "resources": f"{api_config.api_prefix}/resources",
            "frame": f"{api_config.api_prefix}/frame",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Pixelle-Video API Server")
    # In embedded mode we force loopback regardless of what the caller passed,
    # so the default is loopback-only; standalone users still get 0.0.0.0.
    parser.add_argument(
        "--host",
        default="127.0.0.1" if EMBEDDED_MODE else "0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()

    # Defence-in-depth: embedded mode MUST NEVER expose Pixelle beyond the
    # host loopback interface. If the caller tried to override with anything
    # else (e.g. a stale CLI invocation), we coerce back to 127.0.0.1 and
    # emit a warning instead of silently honouring a dangerous bind.
    if EMBEDDED_MODE and args.host not in {"127.0.0.1", "::1", "localhost"}:
        logger.warning(
            "Embedded mode rejected non-loopback host %r; coercing to 127.0.0.1",
            args.host,
        )
        args.host = "127.0.0.1"
    if EMBEDDED_MODE and args.reload:
        # Reload mode spawns a second process that would not inherit the
        # embedded env consistently; disable it to avoid duplicated servers.
        logger.warning("Embedded mode disables --reload; ignoring the flag.")
        args.reload = False
    
    # Print startup banner
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Pixelle-Video API Server                      ║
╚══════════════════════════════════════════════════════════════╝

Starting server at http://{args.host}:{args.port}
API Docs: http://{args.host}:{args.port}/docs
ReDoc: http://{args.host}:{args.port}/redoc

Press Ctrl+C to stop the server
""")
    
    # Start server
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

