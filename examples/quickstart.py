"""Two threads, two users, one store: what crosses threads and what doesn't (offline).

    python -m examples.quickstart                  # fake model, nothing to install
    MEMORY_PROVIDER=ollama python -m examples.quickstart   # the same, on ollama

Swap ``Settings.offline()`` for ``Settings.from_env()`` to run it on your own model.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from memory_agent import Assistant
from memory_agent import Settings
from memory_agent.demo import run_demo


def main() -> int:
    offline = os.environ.get("MEMORY_PROVIDER", "fake") == "fake"
    settings = Settings.offline() if offline else Settings.from_env()
    store = os.environ.get("MEMORY_STORE")  # e.g. s3+file:///tmp/memories (optional adapter)
    with tempfile.TemporaryDirectory() as tmp:
        threads = str(Path(tmp) / "threads.db")
        memories = store or str(Path(tmp) / "memories.db")
        with Assistant.open(settings, threads=threads, memories=memories) as bot:
            checks = run_demo(bot, sys.stdout)
    if offline:  # a real model words things its own way; the fake one is checked exactly
        assert all(checks.values()), checks
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
