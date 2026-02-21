"""Codex CLI integration manager.

This module keeps legacy class names (`ClaudeSDKManager`, `ClaudeResponse`,
`StreamUpdate`) so the rest of the bot can remain unchanged while using Codex
as the execution backend.
"""

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from ..config.settings import Settings
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)

logger = structlog.get_logger()


def find_codex_cli(codex_cli_path: Optional[str] = None) -> Optional[str]:
    """Find Codex CLI in common locations."""
    if (
        codex_cli_path
        and os.path.exists(codex_cli_path)
        and os.access(codex_cli_path, os.X_OK)
    ):
        return codex_cli_path

    env_path = os.environ.get("CODEX_CLI_PATH")
    if env_path and os.path.exists(env_path) and os.access(env_path, os.X_OK):
        return env_path

    from_config = shutil.which("codex")
    if from_config:
        return from_config

    common_paths = [
        "/opt/homebrew/bin/codex",
        "/usr/local/bin/codex",
        "/usr/bin/codex",
    ]
    for candidate in common_paths:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


@dataclass
class ClaudeResponse:
    """Response from CLI backend."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """Streaming update for bot progress messages."""

    type: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ClaudeSDKManager:
    """Manage Codex CLI integration (legacy name kept for compatibility)."""

    def __init__(self, config: Settings):
        self.config = config
        self.codex_cli_path = find_codex_cli(
            config.codex_cli_path or config.claude_cli_path
        )
        if not self.codex_cli_path:
            logger.warning("Codex CLI not found in PATH or common locations")

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
    ) -> ClaudeResponse:
        """Execute prompt via Codex CLI with JSON event streaming."""
        if not self.codex_cli_path:
            raise ClaudeProcessError(
                "Codex CLI not found. Install Codex CLI and ensure `codex` is in PATH, "
                "or set CODEX_CLI_PATH."
            )

        start_time = asyncio.get_event_loop().time()
        cmd = self._build_codex_command(
            prompt=prompt,
            session_id=session_id,
            continue_session=continue_session,
        )

        logger.info(
            "Starting Codex CLI command",
            command=cmd,
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_directory),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        response_chunks: List[str] = []
        tools_used: List[Dict[str, Any]] = []
        active_thread_id: Optional[str] = session_id
        turns = 0

        async def read_stdout() -> None:
            nonlocal active_thread_id, turns
            assert process.stdout is not None
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                stdout_lines.append(line)

                event = self._parse_json_event(line)
                if not event:
                    continue

                event_type = event.get("type")
                if event_type == "thread.started":
                    thread_id = event.get("thread_id")
                    if thread_id:
                        active_thread_id = thread_id
                elif event_type == "turn.completed":
                    turns += 1

                item = event.get("item")
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        response_chunks.append(text.strip())
                        if stream_callback:
                            await stream_callback(
                                StreamUpdate(type="assistant", content=text.strip())
                            )

                elif item_type == "reasoning":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip() and stream_callback:
                        await stream_callback(
                            StreamUpdate(type="assistant", content=text.strip())
                        )

                elif item_type == "command_execution" and event_type == "item.started":
                    command = item.get("command")
                    if isinstance(command, str) and command:
                        tool_call = {"name": "Bash", "input": {"command": command}}
                        tools_used.append(tool_call)
                        if stream_callback:
                            await stream_callback(
                                StreamUpdate(type="assistant", tool_calls=[tool_call])
                            )

        async def read_stderr() -> None:
            assert process.stderr is not None
            while True:
                raw = await process.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    stderr_lines.append(line)

        try:
            await asyncio.wait_for(
                asyncio.gather(read_stdout(), read_stderr(), process.wait()),
                timeout=self.config.claude_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise ClaudeTimeoutError(
                f"Codex CLI timed out after {self.config.claude_timeout_seconds}s"
            ) from e

        duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        stderr_text = "\n".join(stderr_lines).strip()

        if process.returncode != 0:
            error_text = (
                stderr_text
                or "\n".join(stdout_lines[-20:])
                or "Unknown Codex CLI error"
            )
            lowered = error_text.lower()
            if "mcp" in lowered:
                raise ClaudeMCPError(f"MCP server error: {error_text}")
            raise ClaudeProcessError(
                f"Codex CLI error (exit {process.returncode}): {error_text}"
            )

        content = "\n\n".join(chunk for chunk in response_chunks if chunk).strip()
        if not content:
            # Fallback for rare cases where no agent_message is emitted.
            for line in reversed(stdout_lines):
                event = self._parse_json_event(line)
                if not event:
                    continue
                item = event.get("item")
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    content = item["text"].strip()
                    if content:
                        break

        return ClaudeResponse(
            content=content or "(No response from Codex CLI)",
            session_id=active_thread_id or session_id or "",
            cost=0.0,
            duration_ms=duration_ms,
            num_turns=turns or 1,
            tools_used=tools_used,
        )

    def _build_codex_command(
        self,
        prompt: str,
        session_id: Optional[str],
        continue_session: bool,
    ) -> List[str]:
        """Build Codex CLI command for new or resumed turn."""
        if continue_session and session_id:
            cmd = [
                self.codex_cli_path or "codex",
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
            ]
            cmd.extend([session_id, prompt])
            return cmd

        cmd = [
            self.codex_cli_path or "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
        ]
        cmd.extend(self._build_common_codex_flags())
        cmd.append(prompt)
        return cmd

    def _build_common_codex_flags(self) -> List[str]:
        """Build Codex flags used for new turns (`codex exec`)."""
        flags: List[str] = []
        if self.config.codex_model:
            flags.extend(["-m", self.config.codex_model])
        if self.config.codex_profile:
            flags.extend(["--profile", self.config.codex_profile])
        if self.config.sandbox_enabled:
            if self.config.codex_use_full_auto:
                flags.append("--full-auto")
            else:
                flags.extend(["--sandbox", self.config.codex_sandbox_mode])
        if self.config.codex_extra_args:
            flags.extend(self.config.codex_extra_args)
        return flags

    @staticmethod
    def _parse_json_event(line: str) -> Optional[Dict[str, Any]]:
        """Parse one JSONL event line from Codex CLI output."""
        if not line.startswith("{"):
            return None
        try:
            event = json.loads(line)
            return event if isinstance(event, dict) else None
        except json.JSONDecodeError as e:
            raise ClaudeParsingError(f"Failed to decode Codex JSON event: {e}") from e

    def get_active_process_count(self) -> int:
        return 0
