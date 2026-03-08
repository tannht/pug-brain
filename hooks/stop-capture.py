#!/usr/bin/env python3
"""Stop hook — auto-flush memories when a Claude Code session ends.

Reads the session transcript, detects memorable patterns (decisions, errors,
TODOs, insights), and saves them to the neural graph before the session
context is lost.

No external dependencies beyond neural-memory itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("neural_memory.hooks.stop")

# Maximum characters to analyze from the transcript tail
_MAX_TRANSCRIPT_CHARS = 40_000
# Maximum messages to scan from the end of the transcript
_MAX_MESSAGES = 80


def _read_hook_input() -> dict:
    """Read JSON hook input from stdin."""
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_transcript_text(transcript_path: str) -> str:
    """Extract recent text content from transcript JSONL file."""
    path = Path(transcript_path)
    if not path.is_file():
        return ""

    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            recent = all_lines[-_MAX_MESSAGES:]

        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = ""
            if isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "") + "\n"
            elif isinstance(msg.get("message"), dict):
                inner = msg["message"]
                if isinstance(inner.get("content"), str):
                    content = inner["content"]
                elif isinstance(inner.get("content"), list):
                    for block in inner["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "") + "\n"

            if content.strip():
                role = msg.get("role", msg.get("type", ""))
                lines.append(f"[{role}] {content.strip()}")

    except OSError:
        return ""

    text = "\n\n".join(lines)
    if len(text) > _MAX_TRANSCRIPT_CHARS:
        text = text[-_MAX_TRANSCRIPT_CHARS:]
    return text


async def _flush_memories(text: str) -> int:
    """Analyze text and flush detected memories to storage.

    Returns saved count.
    """
    if not text:
        return 0

    from neural_memory.mcp.auto_capture import analyze_text_for_memories
    from neural_memory.safety.sensitive import auto_redact_content

    detected = analyze_text_for_memories(
        text,
        capture_decisions=True,
        capture_errors=True,
        capture_todos=True,
        capture_facts=True,
        capture_insights=True,
        capture_preferences=True,
    )

    if not detected:
        return 0

    emergency_threshold = 0.5
    eligible = [item for item in detected if item["confidence"] >= emergency_threshold]

    if not eligible:
        return 0

    boosted = [{**item, "priority": min(item.get("priority", 5) + 2, 10)} for item in eligible]

    from neural_memory.core.typed_memory import MemoryType, Priority, TypedMemory
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.unified_config import get_shared_storage
    from neural_memory.utils.timeutils import utcnow

    storage = await get_shared_storage()
    brain = await storage.get_brain(storage._current_brain_id or "")
    if not brain:
        return 0

    encoder = MemoryEncoder(storage, brain.config)
    saved = 0

    for item in boosted:
        content = item["content"]
        redacted, _, _ = auto_redact_content(content, min_severity=3)
        mem_type_str = item.get("type", "context")

        try:
            mem_type = MemoryType(mem_type_str)
        except ValueError:
            mem_type = MemoryType.CONTEXT

        priority_val = item.get("priority", 5)
        try:
            mem_priority = Priority(min(max(priority_val, 0), 10))
        except ValueError:
            mem_priority = Priority(5)

        try:
            storage.disable_auto_save()
            result = await encoder.encode(
                content=redacted,
                timestamp=utcnow(),
                tags={"emergency_flush", "session_end"},
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=mem_priority,
                source="stop_hook",
                tags={"emergency_flush", "session_end"},
            )
            await storage.add_typed_memory(typed_mem)
            await storage.batch_save()
            saved += 1
        except Exception:
            logger.debug("Failed to save memory during stop flush", exc_info=True)
            continue

    await storage.close()
    return saved


def main() -> None:
    hook_input = _read_hook_input()
    transcript_path = hook_input.get("transcript_path", "")

    text = _extract_transcript_text(transcript_path)

    try:
        saved = asyncio.run(_flush_memories(text))
    except Exception:
        logger.debug("Stop flush failed", exc_info=True)
        saved = 0

    if saved > 0:
        print(f"\nNeuralMemory: Auto-saved {saved} memories from this session.")
    else:
        print("\nNeuralMemory: Session ended (no new memories detected).")


if __name__ == "__main__":
    main()
