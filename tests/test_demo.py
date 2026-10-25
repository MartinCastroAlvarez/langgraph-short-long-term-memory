"""The two-thread demo, on both stores."""

from __future__ import annotations

import io

from memory_agent import Assistant
from memory_agent import Settings
from memory_agent.demo import run_demo


def test_demo_in_memory() -> None:
    with Assistant.open(Settings.offline()) as bot:
        checks = run_demo(bot)
    assert all(checks.values()), checks


def test_demo_on_sqlite(sqlite_paths: tuple[str, str]) -> None:
    threads, memories = sqlite_paths
    out = io.StringIO()
    with Assistant.open(Settings.offline(), threads=threads, memories=memories) as bot:
        checks = run_demo(bot, out)
    assert all(checks.values()), out.getvalue()
    assert "alice/b > What do you know about me?" in out.getvalue()
