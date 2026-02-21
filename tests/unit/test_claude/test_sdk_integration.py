"""Tests for Codex CLI-backed sdk_integration module."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from src.claude.exceptions import ClaudeProcessError, ClaudeTimeoutError
from src.claude.sdk_integration import ClaudeSDKManager, StreamUpdate
from src.config.settings import Settings


class _FakeStream:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self):
        if self._lines:
            return self._lines.pop(0)
        await asyncio.sleep(0)
        return b""


class _FakeProcess:
    def __init__(self, stdout_lines=None, stderr_lines=None, returncode=0):
        self.stdout = _FakeStream(stdout_lines or [])
        self.stderr = _FakeStream(stderr_lines or [])
        self.returncode = returncode
        self.killed = False

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def config(tmp_path):
    return Settings(
        telegram_bot_token="test:token",
        telegram_bot_username="testbot",
        approved_directory=tmp_path,
        claude_timeout_seconds=2,
    )


@pytest.fixture
def manager(config):
    with patch(
        "src.claude.sdk_integration.find_codex_cli", return_value="/usr/local/bin/codex"
    ):
        return ClaudeSDKManager(config)


class TestCodexCommandBuild:
    def test_build_new_command(self, manager):
        cmd = manager._build_codex_command(
            prompt="hello",
            session_id=None,
            continue_session=False,
        )
        assert cmd[:4] == [
            "/usr/local/bin/codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
        ]
        assert cmd[-1] == "hello"

    def test_build_resume_command(self, manager):
        cmd = manager._build_codex_command(
            prompt="continue",
            session_id="thread-1",
            continue_session=True,
        )
        assert cmd[:5] == [
            "/usr/local/bin/codex",
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
        ]
        assert cmd[-2:] == ["thread-1", "continue"]

    def test_build_resume_command_ignores_exec_only_flags(self, manager):
        manager.config.codex_use_full_auto = False
        manager.config.codex_sandbox_mode = "workspace-write"
        manager.config.codex_model = "gpt-5"
        manager.config.codex_profile = "test-profile"
        manager.config.codex_extra_args = ["--sandbox", "danger-full-access"]

        cmd = manager._build_codex_command(
            prompt="continue",
            session_id="thread-1",
            continue_session=True,
        )

        assert cmd == [
            "/usr/local/bin/codex",
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "thread-1",
            "continue",
        ]


class TestExecuteCommand:
    async def test_execute_success_and_streaming(self, manager):
        updates = []

        async def on_stream(update: StreamUpdate):
            updates.append(update)

        fake = _FakeProcess(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"thread-abc"}\n',
                '{"type":"item.completed","item":{"type":"reasoning","text":"Thinking"}}\n',
                '{"type":"item.started","item":{"type":"command_execution","command":"ls -1"}}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}\n',
                '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            ]
        )

        with patch("asyncio.create_subprocess_exec", return_value=fake):
            response = await manager.execute_command(
                prompt="list files",
                working_directory=Path("/tmp"),
                stream_callback=on_stream,
            )

        assert response.session_id == "thread-abc"
        assert response.content == "Done"
        assert response.num_turns == 1
        assert response.tools_used and response.tools_used[0]["name"] == "Bash"
        assert any(u.type == "assistant" and u.content == "Thinking" for u in updates)
        assert any(u.tool_calls for u in updates)

    async def test_execute_resume_reuses_thread_id(self, manager):
        fake = _FakeProcess(
            stdout_lines=[
                '{"type":"thread.started","thread_id":"thread-existing"}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"Resumed ok"}}\n',
            ]
        )

        with patch("asyncio.create_subprocess_exec", return_value=fake):
            response = await manager.execute_command(
                prompt="continue",
                working_directory=Path("/tmp"),
                session_id="thread-existing",
                continue_session=True,
            )

        assert response.session_id == "thread-existing"
        assert response.content == "Resumed ok"

    async def test_execute_nonzero_exit_raises_process_error(self, manager):
        fake = _FakeProcess(
            stdout_lines=['{"type":"turn.started"}\n'],
            stderr_lines=["fatal: no session\n"],
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=fake):
            with pytest.raises(ClaudeProcessError):
                await manager.execute_command(
                    prompt="continue",
                    working_directory=Path("/tmp"),
                    session_id="bad-thread",
                    continue_session=True,
                )

    async def test_execute_timeout_raises(self, manager):
        class _SlowProcess(_FakeProcess):
            async def wait(self):
                await asyncio.sleep(5)
                return self.returncode

        fake = _SlowProcess(stdout_lines=[], stderr_lines=[])

        with patch("asyncio.create_subprocess_exec", return_value=fake):
            with pytest.raises(ClaudeTimeoutError):
                await manager.execute_command(
                    prompt="slow",
                    working_directory=Path("/tmp"),
                )
            assert fake.killed is True

    async def test_missing_codex_binary_raises(self, config):
        with patch("src.claude.sdk_integration.find_codex_cli", return_value=None):
            mgr = ClaudeSDKManager(config)

        with pytest.raises(ClaudeProcessError, match="Codex CLI not found"):
            await mgr.execute_command(prompt="hi", working_directory=Path("/tmp"))
