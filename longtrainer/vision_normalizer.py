"""Multi-LLM Vision Image Normalizer — P3-6.

Provides:
- Schema normalization: Rewrites image content dicts per provider (OpenAI, Anthropic, Gemini)
- Decompression bomb guard: PIL.Image.MAX_IMAGE_PIXELS ceiling
- One-pass JPEG resize: `downscale_image()` with max_dim + 85% quality
- Dedicated thread pool: VISION_POOL (ThreadPoolExecutor, max_workers=4)
  Never uses asyncio.to_thread() — avoids starving the default pool.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Dedicated Thread Pool ────────────────────────────────────────────────────
# Default executor = min(32, cpu_count+4). On 2 vCPU = 6 threads total.
# 10 concurrent image uploads would saturate it, blocking all other async ops.
VISION_POOL = ThreadPoolExecutor(max_workers=4)


# ─── Decompression Bomb Guard ─────────────────────────────────────────────────

def _configure_pil_safety():
    """Set Pillow's global decompression bomb ceiling.

    A 50KB malicious JPEG can decompress to 30GB of raw pixels,
    bypassing Docker memory limits. This ceiling caps at 16M pixels
    and causes DecompressionBombError before any RAM is allocated.
    """
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 16_000_000  # 4000x4000 = safe max
    except ImportError:
        logger.warning("Pillow not installed — skipping decompression bomb guard.")


# Apply on module load
_configure_pil_safety()


# ─── Image Processing ─────────────────────────────────────────────────────────

def downscale_image(raw_bytes: bytes, max_dim: int = 1568) -> bytes:
    """Downscale image to fit within max_dim longest edge, 85% JPEG quality.

    One-pass resize — no binary search compression loop.
    Pre-encode size cap: if raw_bytes > 3.8MB (Anthropic/Gemini safe limit
    before 33% base64 overhead), resize is aggressively applied.

    Args:
        raw_bytes: Raw image bytes.
        max_dim: Maximum dimension for longest edge (default: 1568).

    Returns:
        Resized JPEG bytes.

    Raises:
        ValueError: If the image is a decompression bomb.
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Image.DecompressionBombError:
        raise ValueError("Image dimensions exceed safe limits (decompression bomb detected).")

    # Convert RGBA/palette to RGB for JPEG compatibility
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def downscale_image_async(raw_bytes: bytes, max_dim: int = 1568) -> bytes:
    """Async wrapper using dedicated VISION_POOL (never default executor).

    Args:
        raw_bytes: Raw image bytes.
        max_dim: Maximum dimension for longest edge.

    Returns:
        Resized JPEG bytes.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(VISION_POOL, downscale_image, raw_bytes, max_dim)


# ─── Provider Schema Normalization ────────────────────────────────────────────

# Max safe raw bytes BEFORE base64 encoding (base64 adds ~33% overhead)
# OpenAI accepts URLs so no byte limit concern on our side.
PROVIDER_MAX_RAW_BYTES = {
    "anthropic": 3_800_000,  # ~5MB after base64
    "google": 3_800_000,
    "gemini": 3_800_000,
}


def normalize_image_messages(
    messages: list[dict],
    provider: str,
    image_bytes_map: Optional[dict[str, bytes]] = None,
) -> list[dict]:
    """Rewrite image content dicts to match the target provider's schema.

    This function is synchronous — call downscale_image_async() separately
    for CPU-bound compression before calling this.

    Args:
        messages: List of message dicts with potential image content.
        provider: LLM provider string ('openai', 'anthropic', 'google', 'gemini').
        image_bytes_map: Optional dict mapping image URLs/paths to raw bytes
            for providers that require base64 (Anthropic, Gemini).

    Returns:
        Messages with image content normalized to provider format.
    """
    provider = provider.lower()
    normalized = []

    for msg in messages:
        content = msg.get("content")

        if not isinstance(content, list):
            normalized.append(msg)
            continue

        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                new_content.append(
                    _normalize_image_part(part, provider, image_bytes_map)
                )
            else:
                new_content.append(part)

        normalized.append({**msg, "content": new_content})

    return normalized


def _normalize_image_part(
    part: dict,
    provider: str,
    image_bytes_map: Optional[dict[str, bytes]] = None,
) -> dict:
    """Convert a single image_url part to the target provider's format."""
    image_url = part.get("image_url", {}).get("url", "")

    if provider == "openai":
        # OpenAI accepts URLs directly
        return {"type": "image_url", "image_url": {"url": image_url}}

    elif provider in ("anthropic",):
        # Anthropic requires base64
        raw_bytes = _resolve_image_bytes(image_url, image_bytes_map)
        if raw_bytes is None:
            logger.warning("Cannot resolve image bytes for Anthropic: %s", image_url[:50])
            return part

        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        }

    elif provider in ("google", "gemini"):
        # Gemini requires inline data
        raw_bytes = _resolve_image_bytes(image_url, image_bytes_map)
        if raw_bytes is None:
            logger.warning("Cannot resolve image bytes for Gemini: %s", image_url[:50])
            return part

        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        return {
            "type": "inline_data",
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64,
            },
        }

    # Unknown provider — pass through unchanged
    return part


def _resolve_image_bytes(
    image_url: str,
    image_bytes_map: Optional[dict[str, bytes]] = None,
) -> Optional[bytes]:
    """Resolve image bytes from a URL or pre-loaded bytes map."""
    # Check pre-loaded bytes first
    if image_bytes_map and image_url in image_bytes_map:
        return image_bytes_map[image_url]

    # Try to read from filesystem
    if image_url.startswith("file://"):
        path = image_url[7:]
    elif not image_url.startswith(("http://", "https://", "data:")):
        path = image_url
    else:
        return None

    try:
        with open(path, "rb") as f:
            return f.read()
    except (OSError, FileNotFoundError):
        return None
