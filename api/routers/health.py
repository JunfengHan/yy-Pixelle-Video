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
Health check and system info endpoints
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from pixelle_video.services.llm_service import _is_embedded_mode

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str = "0.1.0"
    service: str = "Pixelle-Video API"
    # ``embedded`` mirrors the PIXELLE_EMBEDDED_MODE env flag so yyvideoclaw's
    # supervisor can distinguish a standalone Pixelle process (which it should
    # NOT tear down) from one it spawned itself during the startup handshake.
    # Field is optional at the schema level so existing standalone consumers
    # that validate against the public OpenAPI spec see no breaking change.
    embedded: Optional[bool] = None


class CapabilitiesResponse(BaseModel):
    """Capabilities response"""
    success: bool = True
    capabilities: dict


def _build_health_payload() -> HealthResponse:
    """Factor out construction so /health and /version stay in sync."""
    # Only surface the flag when it is actually True — this keeps the
    # standalone response shape visually identical to the previous version
    # for casual consumers (curl, browser); yyvideoclaw checks the field
    # explicitly.
    return HealthResponse(embedded=True if _is_embedded_mode() else None)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns service status and version information.
    """
    return _build_health_payload()


@router.get("/version", response_model=HealthResponse)
async def get_version():
    """
    Get API version
    
    Returns version information.
    """
    return _build_health_payload()
