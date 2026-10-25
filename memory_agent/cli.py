"""``python -m memory_agent "<message>"``: one turn of a chat with short- and long-term memory.

python -m memory_agent --user alice --thread a "Hi, my name is Ada and I live in Lisbon."
python -m memory_agent --user alice --thread b "What do you know about me?"
python -m memory_agent --user alice --memories              # long-term memory as JSON
python -m memory_agent --user alice --thread a --history    # checkpoints of a thread
python -m memory_agent --user alice --thread a --fork <checkpoint_id> "Call me Grace."
python -m memory_agent --user alice --forget-me             # wipe memories + checkpoints
python -m memory_agent --provider fake --demo               # the two-thread demo, offline
python -m memory_agent --mermaid chat                       # print a graph
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import TextIO

from memory_agent.assistant import MAX_MESSAGE_CHARS
from memory_agent.assistant import Assistant
from memory_agent.config import DEFAULT_EMBED_MODELS
from memory_agent.config import DEFAULT_MODELS
from memory_agent.config import EMBEDDERS
from memory_agent.config import PROVIDERS
from memory_agent.config import WRITE_MODES
from memory_agent.config import Settings
from memory_agent.consolidation import build_consolidation_graph
from memory_agent.demo import run_demo
from memory_agent.fake import FakeAssistantModel
from memory_agent.graph import build_chat_graph
from memory_agent.llm import ModelConfigError
from memory_agent.namespaces import InvalidIdError
from memory_agent.stores import StoreConfigError

__all__ = ["main", "mermaid"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m memory_agent", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("message", nargs="?", help="what the user says")
    p.add_argument("--user", default="me", help="user id (namespace of long-term memory)")
    p.add_argument("--thread", default="main", help="thread id (short-term memory)")
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--model", help="chat model id for the provider")
    p.add_argument("--embeddings", choices=EMBEDDERS)
    p.add_argument("--write-mode", choices=WRITE_MODES)
    p.add_argument(
        "--data-dir",
        default=".memory",
        help="folder for threads.db (checkpointer) and memories.db (store)",
    )
    p.add_argument("--store", help="long-term store URL instead of <data-dir>/memories.db")
    p.add_argument("--memories", action="store_true", help="print the user's long-term memory")
    p.add_argument("--history", action="store_true", help="print the thread's checkpoints")
    p.add_argument("--replay", metavar="CHECKPOINT", help="re-run from a checkpoint")
    p.add_argument("--fork", metavar="CHECKPOINT", help="send MESSAGE from a checkpoint")
    p.add_argument("--forget-me", action="store_true", help="delete all of the user's memory")
    p.add_argument("--demo", action="store_true", help="run the two-thread demo")
    p.add_argument("--mermaid", choices=("chat", "consolidate"), help="print a graph and exit")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    changes: dict[str, Any] = {}
    if args.provider:
        changes["provider"] = args.provider
        changes["model"] = args.model or DEFAULT_MODELS[args.provider]
        if args.provider == "fake" and not args.embeddings:
            args.embeddings = "hashing"
    elif args.model:
        changes["model"] = args.model
    if args.embeddings:
        changes["embeddings"] = args.embeddings
        changes["embed_model"], changes["embed_dims"] = DEFAULT_EMBED_MODELS[args.embeddings]
    if args.write_mode:
        changes["write_mode"] = args.write_mode
    return dataclasses.replace(base, **changes)


def mermaid(name: str) -> str:
    settings = Settings.offline()
    model = FakeAssistantModel()
    graph = (
        build_chat_graph(settings, model)
        if name == "chat"
        else build_consolidation_graph(settings, model)
    )
    return graph.get_graph().draw_mermaid()


def _paths(args: argparse.Namespace) -> tuple[str, str]:
    if args.demo and not args.store and args.data_dir == ".memory":
        return ":memory:", ":memory:"
    folder = Path(args.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / "threads.db"), args.store or str(folder / "memories.db")


def _emit(data: Any, args: argparse.Namespace, out: TextIO) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str), file=out)


def _run(bot: Assistant, args: argparse.Namespace, out: TextIO) -> int:
    if args.demo:
        checks = run_demo(bot, None if args.json else out)
        if args.json:
            _emit({"passed": all(checks.values()), "checks": checks}, args, out)
        return 0 if all(checks.values()) else 1
    if args.forget_me:
        _emit(bot.forget_user(args.user), args, out)
        return 0
    if args.memories:
        _emit(bot.memories(args.user), args, out)
        return 0
    if args.history:
        rows = [dataclasses.asdict(c) for c in bot.history(args.user, args.thread)]
        if args.json:
            _emit(rows, args, out)
        else:
            for r in rows:
                nxt = ",".join(r["next"]) or "-"
                print(
                    f"{r['checkpoint_id']}  step={r['step']:>3}  next={nxt:<10} "
                    f"messages={r['messages']:<3} {r['last']}",
                    file=out,
                )
        return 0
    if args.replay:
        reply = bot.replay(args.user, args.thread, args.replay)
    elif args.message is None:
        _parser().print_usage(sys.stderr)
        return 2
    elif len(args.message) > MAX_MESSAGE_CHARS:
        print(f"the message is longer than {MAX_MESSAGE_CHARS} characters", file=sys.stderr)
        return 2
    elif args.fork:
        reply = bot.fork(args.user, args.thread, args.fork, args.message)
    else:
        reply = bot.chat(args.user, args.thread, args.message)
    if args.json:
        _emit(dataclasses.asdict(reply), args, out)
    else:
        print(reply.text, file=out)
        if reply.tool_calls:
            print(f"  [tools: {', '.join(reply.tool_calls)}]", file=out)
        if reply.summarized:
            print(f"  [summarized {reply.summarized} older messages]", file=out)
    return 0


def main(argv: Sequence[str] | None = None, out: TextIO = sys.stdout) -> int:
    args = _parser().parse_args(argv)
    if args.mermaid:
        print(mermaid(args.mermaid), file=out)
        return 0
    try:
        settings = _settings(args)
        threads, memories = _paths(args)
        with Assistant.open(settings, threads=threads, memories=memories) as bot:
            return _run(bot, args, out)
    except (ValueError, KeyError, InvalidIdError, ModelConfigError, StoreConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
