# PugBrain — Instructions for Claude Code

> Copy this section into your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` (global).

## Memory System

This workspace uses **PugBrain** for persistent memory across sessions.
You have access to `pugbrain_*` MCP tools. Use them **proactively** — do not wait for the user to ask.

### Session Start (ALWAYS do this)

```
pugbrain_recap()                          # Resume context from last session
pugbrain_context(limit=20, fresh_only=true)  # Load recent memories
pugbrain_session(action="get")            # Check current task/feature/progress
```

If `gap_detected: true`, run `pugbrain_auto(action="flush", text="<recent context>")` to recover lost content.

### During Work — REMEMBER automatically

| Event | Action |
|-------|--------|
| Decision made | `pugbrain_remember(content="...", type="decision", priority=7)` |
| Bug fixed | `pugbrain_remember(content="...", type="error", priority=7)` |
| User preference stated | `pugbrain_remember(content="...", type="preference", priority=6)` |
| Important fact learned | `pugbrain_remember(content="...", type="fact", priority=5)` |
| TODO identified | `pugbrain_todo(task="...", priority=6)` |
| Workflow discovered | `pugbrain_remember(content="...", type="workflow", priority=6)` |

### During Work — RECALL before asking

Before asking the user a question, check memory first:

```
pugbrain_recall(query="<topic>", depth=1)
```

Depth guide: 0=instant lookup, 1=context (default), 2=patterns, 3=deep graph traversal.

### Session End / Before Compaction

```
pugbrain_auto(action="process", text="<summary of session>")
pugbrain_session(action="set", feature="...", task="...", progress=0.8)
```

Before `/compact` or `/new`:
```
pugbrain_auto(action="flush", text="<recent conversation>")
```

### Project Context

```
pugbrain_eternal(action="save", project_name="MyProject", tech_stack=["React", "Node.js"])
pugbrain_eternal(action="save", decision="Use PostgreSQL", reason="Team expertise")
```

### Codebase Indexing

First time on a project:
```
pugbrain_index(action="scan", path="./src")
```

Then `pugbrain_recall(query="authentication")` finds related code through the neural graph.

### Knowledge Base Training

Train permanent knowledge from documentation files:
```
# Train from docs directory (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD, TXT, RST)
pugbrain_train(action="train", path="docs/", domain_tag="react")

# Train a single file
pugbrain_train(action="train", path="api-spec.pdf")

# Check training status
pugbrain_train(action="status")
```

Trained knowledge is **pinned** — permanent, no decay, no pruning. Re-training same file is skipped (SHA-256 dedup).

For non-text formats: `pip install neural-memory[extract]`

### Pin/Unpin Memories

```
pugbrain_pin(fiber_ids=["id1", "id2"], pinned=true)   # Make permanent
pugbrain_pin(fiber_ids=["id1"], pinned=false)           # Resume lifecycle
```

### Health & Diagnostics

```
pugbrain_health()                              # Brain health score + warnings
pugbrain_stats()                               # Memory counts and freshness
pugbrain_alerts(action="list")                 # Active health alerts
pugbrain_conflicts(action="list")              # Conflicting memories
pugbrain_evolution()                           # Brain maturation + plasticity
```

### Spaced Repetition

```
pugbrain_review(action="queue")                # Get memories due for review
pugbrain_review(action="mark", fiber_id="...", success=true)  # Record result
```

### Brain Versioning & Transplant

```
pugbrain_version(action="create", name="pre-refactor")  # Snapshot
pugbrain_version(action="rollback", version_id="...")    # Restore
pugbrain_transplant(source_brain="other", tags=["react"])  # Import from another brain
```

### Multi-Device Sync

```
pugbrain_sync(action="full")                   # Bi-directional sync with hub
pugbrain_sync_status()                         # Check sync status
pugbrain_sync_config(action="set", hub_url="https://hub:8080", enabled=true)
```

### Import External Data

```
pugbrain_import(source="chromadb", connection="/path/to/chroma")
pugbrain_import(source="mem0", user_id="user123")
```

### Edit & Forget — Correct Mistakes

```
# Fix wrong type (auto-detector got it wrong)
pugbrain_edit(memory_id="fiber-abc", type="insight")

# Fix wrong content
pugbrain_edit(memory_id="fiber-abc", content="Corrected: the bug was in auth.py, not login.py")

# Adjust priority
pugbrain_edit(memory_id="fiber-abc", priority=9)

# Soft delete — memory decays naturally (recommended for outdated info)
pugbrain_forget(memory_id="fiber-abc", reason="outdated")

# Hard delete — permanent removal (for sensitive data, test garbage)
pugbrain_forget(memory_id="fiber-abc", hard=true)
```

### Cognitive Reasoning

```
pugbrain_hypothesize(action="create", content="Redis is the bottleneck", confidence=0.6)
pugbrain_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")
pugbrain_predict(action="create", content="Fix will drop latency 50%", hypothesis_id="h-1", deadline="2026-04-01")
pugbrain_verify(prediction_id="p-1", outcome="correct")
pugbrain_cognitive(action="summary")           # Hot index
pugbrain_gaps(action="detect", topic="...")    # Track unknowns
pugbrain_schema(action="evolve", hypothesis_id="h-1", content="...", reason="...")
```

### Connection Tracing

```
pugbrain_explain(entity_a="Redis", entity_b="auth outage")
```

Traces shortest path with evidence. Use to debug recall or verify connections.

### Rules

1. **Be proactive** — remember important info without being asked
2. **Store 3-5 memories per task** — a bug fix has: root cause, fix, insight, prevention
3. **Use rich language** — "Chose X over Y because Z" not just "X". Mix causal, temporal, relational, comparative
4. **Check memory first** — recall before asking questions the user may have answered before
5. **Use diverse types** — fact, decision, error, preference, todo, workflow, insight, instruction, context
6. **Set priority** — critical=7-10, normal=5, trivial=1-3
7. **Add tags** — always include project name + topic for better retrieval
8. **Recap on start** — always call `pugbrain_recap()` at session beginning
9. **Train KB first** — if project has docs/, train them into memory for permanent context
10. **Fix mistakes** — use `pugbrain_edit` for wrong types/content, `pugbrain_forget` for outdated info
11. **Health weekly** — `pugbrain_health()` and fix the highest penalty first
