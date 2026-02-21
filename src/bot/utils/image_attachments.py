"""Utilities for attaching generated image artifacts to Telegram replies.

Scans the active working directory for image files created/modified during a
Claude run, then uploads those files back to the chat.
"""

import os
from pathlib import Path
from typing import List, Tuple

import structlog
from telegram import Message

from .html_format import escape_html

logger = structlog.get_logger()

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}

_TELEGRAM_PHOTO_MAX_BYTES = 10 * 1024 * 1024


def _find_recent_image_files(
    working_directory: Path,
    started_at_ts: float,
    max_files: int,
) -> Tuple[List[Path], int]:
    """Return recent image files under working_directory.

    The list includes files with supported extensions whose mtime is newer than
    started_at_ts. Returns (files_to_send, total_detected).
    """
    if not working_directory.exists() or not working_directory.is_dir():
        return [], 0

    candidates: list[tuple[float, Path]] = []

    for root, dirs, files in os.walk(working_directory):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS and not d.startswith(".")]

        root_path = Path(root)
        for filename in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in _IMAGE_EXTENSIONS:
                continue

            file_path = root_path / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue

            if stat.st_size <= 0:
                continue
            if stat.st_mtime < started_at_ts:
                continue

            candidates.append((stat.st_mtime, file_path))

    candidates.sort(key=lambda item: item[0])
    total_detected = len(candidates)
    recent_files = [path for _, path in candidates[-max_files:]]
    return recent_files, total_detected


async def send_recent_generated_images(
    message: Message,
    working_directory: Path,
    started_at_ts: float,
    max_files: int = 5,
) -> int:
    """Upload recent generated image files and return number sent."""
    files_to_send, total_detected = _find_recent_image_files(
        working_directory=working_directory,
        started_at_ts=started_at_ts,
        max_files=max_files,
    )
    if not files_to_send:
        return 0

    sent_count = 0
    for index, file_path in enumerate(files_to_send):
        try:
            file_size = file_path.stat().st_size
            caption = f"🖼️ Generated image: <code>{escape_html(file_path.name)}</code>"
            reply_to_id = message.message_id if index == 0 else None

            with file_path.open("rb") as image_file:
                if file_size <= _TELEGRAM_PHOTO_MAX_BYTES:
                    await message.reply_photo(
                        photo=image_file,
                        caption=caption,
                        parse_mode="HTML",
                        reply_to_message_id=reply_to_id,
                    )
                else:
                    await message.reply_document(
                        document=image_file,
                        filename=file_path.name,
                        caption=caption,
                        parse_mode="HTML",
                        reply_to_message_id=reply_to_id,
                    )
            sent_count += 1
        except Exception as exc:
            logger.warning(
                "Failed to upload generated image",
                file_path=str(file_path),
                error=str(exc),
            )

    if total_detected > max_files:
        try:
            await message.reply_text(
                f"ℹ️ Sent {sent_count} recent images (showing latest {max_files} of "
                f"{total_detected} detected)."
            )
        except Exception as exc:
            logger.warning("Failed to send image summary message", error=str(exc))

    return sent_count
