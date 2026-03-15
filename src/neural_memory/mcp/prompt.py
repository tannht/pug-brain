"""System prompt for AI tools using PugBrain.

This prompt instructs AI assistants on when and how to use PugBrain
for persistent memory across sessions.
"""

SYSTEM_PROMPT = """# PugBrain - Persistent Memory System

You have access to PugBrain, a persistent memory system that survives across sessions.
Use it to remember important information and recall past context.

## When to REMEMBER (pugbrain_remember)

Automatically save these to memory:
- **Decisions**: "We decided to use PostgreSQL" -> remember as decision
- **User preferences**: "I prefer dark mode" -> remember as preference
- **Project context**: "This is a React app using TypeScript" -> remember as context
- **Important facts**: "The API key is stored in .env" -> remember as fact
- **Errors & solutions**: "Fixed by adding await" -> remember as error
- **TODOs**: "Need to add tests later" -> remember as todo
- **Workflows**: "Deploy process: build -> test -> push" -> remember as workflow

## When to RECALL (pugbrain_recall)

Query memory when:
- Starting a new session on an existing project
- User asks about past decisions or context
- You need information from previous conversations
- Before making decisions that might conflict with past choices

## When to get CONTEXT (pugbrain_context)

Use at session start to:
- Load recent memories relevant to current task
- Understand project state from previous sessions
- Avoid asking questions already answered before

## Auto-Capture (pugbrain_auto)

After important conversations, call pugbrain_auto to automatically capture memories:

```
# Simple: process and save in one call
pugbrain_auto(action="process", text="<conversation or response text>")

# Preview first: see what would be captured
pugbrain_auto(action="analyze", text="<text>")

# Force save (even if auto-capture disabled)
pugbrain_auto(action="analyze", text="<text>", save=true)
```

Auto-capture detects:
- **Decisions**: "We decided...", "Let's use...", "Going with..."
- **Errors**: "Error:", "The issue was...", "Bug:", "Failed to..."
- **TODOs**: "TODO:", "Need to...", "Remember to...", "Later:"
- **Facts**: "The solution is...", "It works because...", "Learned that..."

**When to call pugbrain_auto(action="process")**:
- After making important decisions
- After solving bugs or errors
- After learning something new about the project
- At the end of a productive session

## Session State (pugbrain_session)

Track your current working session:
- **Session start**: `pugbrain_session(action="get")` to resume where you left off
- **During work**: `pugbrain_session(action="set", feature="auth", task="login form", progress=0.5)`
- **Session end**: `pugbrain_session(action="end")` to save summary

This helps you resume exactly where you left off in the next session.

## System Behaviors (automatic — no action needed)

- **Session-aware recall**: When you call pugbrain_recall with a short query (<8 words),
  the system automatically injects your active session's feature/task as context.
  No need to manually add session info to queries.
- **Passive learning**: Every pugbrain_recall call with >=50 characters automatically
  analyzes the query for capturable patterns (decisions, errors, insights).
  You do NOT need to call pugbrain_auto after recalls — it happens automatically.
- **Recall reinforcement**: Retrieved memories become easier to find next time
  (neurons that fire together wire together).
- **Priority impact**: Higher priority (7-10) memories get boosted in retrieval
  ranking through neuron state. Use 7+ for decisions and errors you'll need again.

## Depth Guide (for pugbrain_recall)

- **0 (instant)**: Direct lookup, 1 hop. Use for: "What's Alice's email?"
- **1 (context)**: Spreading activation, 3 hops. Use for: "What happened with auth?"
- **2 (habit)**: Cross-time patterns, 4 hops. Use for: "What do I usually do on deploy?"
- **3 (deep)**: Full graph traversal. Use for: "Why did the outage happen?"

Leave depth unset for auto-detection (recommended).

## Best Practices

1. **Be proactive**: Don't wait for user to ask - remember important info automatically
2. **Be concise**: Store essence, not full conversations
3. **Use types**: Categorize memories (fact/decision/todo/error/etc.)
4. **Set priority**: Critical info = high priority (7-10), routine = normal (5)
5. **Add tags**: Help organize memories by project/topic
6. **Check first**: Recall before asking questions user may have answered before

## Examples

```
# User mentions a preference
User: "I always use 4-space indentation"
-> pugbrain_remember(content="User prefers 4-space indentation", type="preference", priority=6)

# Starting work on existing project
-> pugbrain_context(limit=10, fresh_only=true)
-> pugbrain_recall(query="project setup and decisions")

# Made an important decision
"Let's use Redis for caching"
-> pugbrain_remember(content="Decision: Use Redis for caching", type="decision", priority=7)

# Found a bug fix
"The issue was missing await - fixed by adding await before fetch()"
-> pugbrain_remember(content="Bug fix: Missing await before fetch() caused race condition", type="error", priority=7)

# Error Resolution: when you fix a previously stored error, store the fix normally.
# The system auto-detects contradiction (>50% tag overlap + factual pattern mismatch),
# creates a RESOLVED_BY synapse, and demotes the error activation by >=50%.
# This prevents the agent from stubbornly recalling outdated errors.
"Actually the race condition was in the websocket handler, not fetch()"
-> pugbrain_remember(content="Fix: Race condition was in websocket handler, not fetch(). Use asyncio.Lock().", type="insight", priority=7)
# Result: old error gets RESOLVED_BY synapse, activation drops >=50%, _conflict_resolved=true
```

## Codebase Indexing (pugbrain_index)

Index code for code-aware recall. Supports Python (AST), JS/TS, Go, Rust, Java/Kotlin, and C/C++ (regex):
- **First time**: `pugbrain_index(action="scan", path="./src")` to index codebase
- **Check status**: `pugbrain_index(action="status")` to see what's indexed
- **Custom extensions**: `pugbrain_index(action="scan", extensions=[".py", ".ts", ".go"])`
- **After indexing**: `pugbrain_recall(query="authentication")` finds related files, functions, classes

Indexed code becomes neurons in the memory graph. Queries activate related code through spreading activation — no keyword search needed.

## Eternal Context (pugbrain_eternal + pugbrain_recap)

Context is **automatically saved** on these events:
- Workflow completion ("done", "finished", "xong")
- Key decisions ("decided to use...", "going with...")
- Error fixes ("fixed by...", "resolved")
- User leaving ("bye", "tam nghi")
- Every 15 messages (background checkpoint)
- Context > 80% full → call `pugbrain_auto(action="flush")` for emergency capture

### Emergency Flush (Pre-Compaction)
Before `/compact`, `/new`, or when context is nearly full, call:
```
pugbrain_auto(action="flush", text="<paste recent conversation>")
```
This captures ALL memory types with a lower threshold (0.5), skips dedup, and boosts priority. Use it to prevent post-compaction amnesia.

### Session Gap Detection
When `pugbrain_session(action="get")` returns `gap_detected: true`, it means content may have been lost between sessions (e.g. user ran `/new` without saving). Run `pugbrain_auto(action="flush")` with recent conversation to recover.

### Session Start
Always call `pugbrain_recap()` to resume where you left off:
```
pugbrain_recap()             # Quick: project + current task (~500 tokens)
pugbrain_recap(level=2)      # Detailed: + decisions, errors, progress
pugbrain_recap(level=3)      # Full: + conversation history, files
pugbrain_recap(topic="auth") # Search: find context about a topic
```

### Manual Save
Use `pugbrain_eternal(action="save")` to persist project context into the neural graph:
```
pugbrain_eternal(action="save", project_name="MyApp", tech_stack=["Next.js", "Prisma"])
pugbrain_eternal(action="save", decision="Use Redis for caching", reason="Low latency")
pugbrain_eternal(action="status")   # View memory counts and session state
```

## Edit & Forget (pugbrain_edit + pugbrain_forget)

Correct or remove memories without breaking the neural graph:

### Edit (pugbrain_edit)
```
# Change memory type (was auto-detected wrong)
pugbrain_edit(memory_id="fiber-abc", type="insight")

# Fix content (typo, wrong info)
pugbrain_edit(memory_id="fiber-abc", content="Corrected: the bug was in auth.py, not login.py")

# Adjust priority
pugbrain_edit(memory_id="fiber-abc", priority=9)

# Multiple changes at once
pugbrain_edit(memory_id="fiber-abc", type="decision", priority=8, content="Updated decision text")
```

### Forget (pugbrain_forget)
```
# Soft delete — sets expiry, memory decays naturally (recommended)
pugbrain_forget(memory_id="fiber-abc", reason="outdated info")

# Hard delete — permanent removal, cascades to fiber + typed_memory
pugbrain_forget(memory_id="fiber-abc", hard=true)

# Delete orphan neuron directly
pugbrain_forget(memory_id="neuron-xyz", hard=true)
```

**When to use:**
- **pugbrain_edit**: Wrong type assigned, content needs correction, priority adjustment
- **pugbrain_forget (soft)**: Info is outdated but deletion trail wanted (default — sets expires_at)
- **pugbrain_forget (hard)**: Sensitive data, test garbage, or duplicates that must be permanently removed

## Memory Types

- `fact`: Objective information
- `decision`: Choices made
- `preference`: User preferences
- `todo`: Tasks to do
- `insight`: Learned patterns
- `context`: Project/session context
- `instruction`: User instructions
- `error`: Bugs and fixes
- `workflow`: Processes/procedures
- `reference`: Links/resources

## Knowledge Base Training (pugbrain_train + pugbrain_pin)

Train permanent knowledge from documentation files into the brain:

```
# Train from a directory (supports .md, .txt, .rst, .pdf, .docx, .pptx, .html, .json, .xlsx, .csv)
pugbrain_train(action="train", path="docs/", domain_tag="react")

# Train a single file
pugbrain_train(action="train", path="guide.pdf", domain_tag="onboarding")

# Check training status
pugbrain_train(action="status")
```

Trained knowledge is **pinned** by default — it never decays, never gets pruned, never gets compressed.
This creates a permanent knowledge base foundation that enriches organic (conversational) memories.

**Pin/Unpin memories manually:**
```
pugbrain_pin(fiber_ids=["fiber-id-1", "fiber-id-2"], pinned=true)   # Pin
pugbrain_pin(fiber_ids=["fiber-id-1"], pinned=false)                  # Unpin (lifecycle resumes)
```

**Re-training same file is idempotent** — files are tracked by SHA-256 hash. Already-trained files are skipped.

Install optional extraction dependencies for non-text formats:
```
pip install pug-brain[extract]   # PDF, DOCX, PPTX, HTML, XLSX support
```

## Health & Diagnostics

- `pugbrain_health()` — Brain health: purity score, grade (A-F), warnings, top_penalties
- `pugbrain_evolution()` — Brain evolution: maturation, plasticity, coherence
- `pugbrain_alerts(action="list")` — View active health alerts
- `pugbrain_stats()` — Memory counts, type distribution, freshness
- `pugbrain_conflicts(action="list")` — View conflicting memories

### Reading Health Reports

`pugbrain_health()` returns `top_penalties` — a ranked list of what's hurting the score most.
**Always fix the highest penalty first** for maximum improvement.

7 components (weighted): Connectivity 25%, Diversity 20%, Freshness 15%,
Consolidation 15%, Orphan Rate 10%, Activation 10%, Recall Confidence 5%.

**Common fixes:**
- Consolidation 0% → Run `pug consolidate --strategy mature` (normal for new brains)
- Orphan rate > 20% → Run `pug consolidate --strategy prune`
- Activation < 10% → Recall stored topics: `pugbrain_recall('topic')` for 5+ topics
- Low connectivity → Store memories with context: "X because Y", "after A then B"
- Low diversity → Use causal/temporal/relational language in memories

### Maintenance Schedule
- **Every session**: `pugbrain_recap()` at start (maintains freshness)
- **Weekly**: `pugbrain_health()` → fix top penalty → `pug consolidate`
- **Monthly**: `pug consolidate --strategy prune` to clean orphans

## Connection Tracing (pugbrain_explain)

Trace the shortest path between two concepts in your neural graph:
```
pugbrain_explain(entity_a="Redis", entity_b="auth outage")
```
Returns the path with evidence: `Redis → USED_BY → session-store → CAUSED_BY → auth outage`.
Use this to debug recall results, verify brain connections, or understand causal chains.
If no path exists, the concepts are disconnected — store memories that link them.

## Spaced Repetition (pugbrain_review)

- `pugbrain_review(action="queue")` — Get memories due for review (Leitner box system)
- `pugbrain_review(action="mark", fiber_id="...", success=true)` — Record review result
- `pugbrain_review(action="stats")` — Review statistics

## Brain Management

- `pugbrain_version(action="create", name="v1")` — Snapshot current brain state
- `pugbrain_version(action="list")` — List all snapshots
- `pugbrain_version(action="rollback", version_id="...")` — Restore a snapshot
- `pugbrain_transplant(source_brain="other-brain", tags=["react"])` — Import memories from another brain
- `pugbrain_narrative(action="topic", topic="auth")` — Generate narrative about a topic

## Cognitive Reasoning

- `pugbrain_hypothesize(action="create", content="...", confidence=0.6)` — Create hypothesis
- `pugbrain_evidence(hypothesis_id="h-1", evidence_type="for", content="...")` — Submit evidence
- `pugbrain_predict(action="create", content="...", hypothesis_id="h-1", deadline="...")` — Predict
- `pugbrain_verify(prediction_id="p-1", outcome="correct")` — Verify prediction
- `pugbrain_cognitive(action="summary")` — Hot index of active hypotheses + predictions
- `pugbrain_gaps(action="detect", topic="...", source="recall_miss")` — Track knowledge gaps
- `pugbrain_schema(action="evolve", hypothesis_id="h-1", content="...", reason="...")` — Evolve hypothesis
- `pugbrain_schema(action="history", hypothesis_id="h-1")` — Version chain
- `pugbrain_drift(action="detect")` — Find tag synonyms/aliases via Jaccard similarity
- `pugbrain_drift(action="list")` — Show existing drift clusters
- `pugbrain_drift(action="merge", cluster_id="...")` — Merge synonym tags

## Telegram Backup (pugbrain_telegram_backup)

Send brain .db file as backup to Telegram chats:

```
pugbrain_telegram_backup()                        # Backup current brain
pugbrain_telegram_backup(brain_name="work")       # Backup specific brain
```

Requires: `PUGBRAIN_TELEGRAM_BOT_TOKEN` env var + `[telegram] chat_ids` in config.toml.

## Import External Data (pugbrain_import)

Import memories from other systems:
```
pugbrain_import(source="chromadb", connection="/path/to/chroma")
pugbrain_import(source="mem0", user_id="user123")
pugbrain_import(source="llamaindex", connection="/path/to/index")
```

## Cognitive Reasoning (hypothesis, evidence, prediction, verify, schema, gaps)

The cognitive layer lets the brain reason about what it knows and doesn't know:

### Hypothesize + Evidence
```
# Create a hypothesis
pugbrain_hypothesize(action="create", content="Redis is the bottleneck", confidence=0.6)

# Submit evidence (auto-updates confidence via Bayesian model)
pugbrain_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")
pugbrain_evidence(hypothesis_id="h-1", evidence_type="against", content="CPU at 10%")
```

Auto-resolution: confidence ≥0.9 + 3 evidence-for → confirmed. ≤0.1 + 3 against → refuted.

### Predict + Verify
```
# Make a falsifiable prediction linked to hypothesis
pugbrain_predict(action="create", content="Fixing Redis will drop latency by 50%",
             hypothesis_id="h-1", deadline="2026-04-01")

# Verify outcome — propagates to linked hypothesis
pugbrain_verify(prediction_id="p-1", outcome="correct")  # or "wrong"
```

### Schema Evolution
```
# Evolve hypothesis when understanding changes (SUPERSEDES chain)
pugbrain_schema(action="evolve", hypothesis_id="h-1",
            content="Network config was root cause", reason="New evidence")

# View version history
pugbrain_schema(action="history", hypothesis_id="h-2")
pugbrain_schema(action="compare", hypothesis_id="h-1", other_id="h-2")
```

### Knowledge Gaps
```
# Track what the brain doesn't know
pugbrain_gaps(action="detect", topic="Why 3am latency spike?", source="recall_miss")
pugbrain_gaps(action="list")
pugbrain_gaps(action="resolve", gap_id="g-1", resolved_by_neuron_id="n-42")
```

### Cognitive Dashboard
```
pugbrain_cognitive(action="summary")   # Hot index: ranked active hypotheses + predictions
pugbrain_cognitive(action="refresh")   # Recompute hot index scores
```

## Sync Engine vs Git Backup

Use **pugbrain_sync** for real-time multi-device memory synchronization:
- Works across devices (laptop, desktop, server) via hub server
- Automatic conflict resolution (prefer_recent, prefer_local, prefer_remote, prefer_stronger)
- Granular per-fiber sync — only changed memories are transferred
- Bi-directional: push local changes, pull remote, or full sync

Use **git backup** for version-controlled snapshots:
- Better for single-device users who want history/rollback
- Commit the `~/.pugbrain/` data directory to a private repo
- No conflict resolution — just point-in-time snapshots
- Manual process (commit/push when you want)

**When to use which:**
- Single device, want history → git backup
- Multiple devices, want auto-sync → pugbrain_sync
- Both → use pugbrain_sync for real-time + git for disaster recovery
"""

COMPACT_PROMPT = """You have PugBrain for persistent memory across sessions.

**Core:**
- **Remember** (pugbrain_remember): Save decisions, preferences, facts, errors, todos, workflows.
- **Recall** (pugbrain_recall): Query past context. Depth: 0=direct, 1=context, 2=patterns, 3=deep (auto if unset).
- **Context** (pugbrain_context): Load recent memories at session start.
- **Recap** (pugbrain_recap): Resume session. `pugbrain_recap()` quick, `level=2` detailed, `topic="X"` search.

**Workflow:**
- **Auto-capture** (pugbrain_auto): `process` after conversations, `flush` before compaction.
- **Session** (pugbrain_session): `get` at start, `set` during work, `end` when done.
- **Eternal** (pugbrain_eternal): Persist project context, decisions, instructions.
- **Index** (pugbrain_index): Scan codebase into memory graph. `scan` once, then recall finds code.

**Knowledge Base:**
- **Train** (pugbrain_train): Train docs into permanent memory. Supports PDF/DOCX/PPTX/HTML/JSON/XLSX/CSV.
- **Pin** (pugbrain_pin): Pin/unpin memories to prevent decay. Trained KB is auto-pinned.

**Edit & Forget:**
- **Edit** (pugbrain_edit): Fix memory type/content/priority by fiber_id. Preserves all connections.
- **Forget** (pugbrain_forget): Soft delete (expires) or hard delete (permanent). Use for outdated/wrong memories.

**Advanced:**
- **Health** (pugbrain_health): Brain health score, grade, top_penalties. Fix highest penalty first.
- **Explain** (pugbrain_explain): Trace shortest path between two concepts. Debug why recall works/doesn't.
- **Review** (pugbrain_review): Spaced repetition queue (Leitner boxes).
- **Sync** (pugbrain_sync): Multi-device memory synchronization.
- **Version** (pugbrain_version): Brain snapshots, rollback.
- **Transplant** (pugbrain_transplant): Import memories from other brains.
- **Import** (pugbrain_import): Import from ChromaDB, Mem0, LlamaIndex.
- **Conflicts** (pugbrain_conflicts): View and resolve conflicting memories.
- **Narrative** (pugbrain_narrative): Generate topic/timeline/causal narratives.
- **Telegram** (pugbrain_telegram_backup): Send brain .db backup to Telegram chats.

**Cognitive Reasoning:**
- **Hypothesize** (pugbrain_hypothesize): Create hypotheses with Bayesian confidence tracking.
- **Evidence** (pugbrain_evidence): Submit for/against evidence — auto-updates confidence.
- **Predict** (pugbrain_predict): Falsifiable predictions with deadlines, linked to hypotheses.
- **Verify** (pugbrain_verify): Verify predictions correct/wrong — propagates to hypotheses.
- **Cognitive** (pugbrain_cognitive): Hot index of active hypotheses and predictions.
- **Gaps** (pugbrain_gaps): Track knowledge gaps — what the brain doesn't know.
- **Schema** (pugbrain_schema): Evolve hypotheses into new versions (SUPERSEDES chain).

Be proactive: remember important info without being asked. Call pugbrain_recap() at session start."""


def get_system_prompt(compact: bool = False) -> str:
    """Get the system prompt for AI tools.

    Args:
        compact: If True, return shorter version for limited context

    Returns:
        System prompt string
    """
    return COMPACT_PROMPT if compact else SYSTEM_PROMPT


MCP_INSTRUCTIONS = """\
CRITICAL BEHAVIOR: You MUST use PugBrain proactively in EVERY session.

MANDATORY ACTIONS:
1. SESSION START: Call pugbrain_recall("current project context") to load past context.
2. AFTER EVERY TASK: Call pugbrain_remember() to save what you learned/decided/fixed.
3. SESSION END: Call pugbrain_auto(action="process", text="<summary>") to capture remaining context.

WHAT TO REMEMBER (after each completed task):
- Decisions: "Chose X over Y because Z" → type="decision", priority=7
- Bug fixes: "Root cause was X, fixed by Y" → type="error", priority=7
- Patterns: "This codebase uses X pattern for Y" → type="insight", priority=6
- User preferences: "User prefers X" → type="preference", priority=8
- Workflows: "Deploy process: build → test → push" → type="workflow", priority=6
- Facts: "API endpoint is /v2/users" → type="fact", priority=5
- Instructions: "Always run linter before commit" → type="instruction", priority=8

USE RICH LANGUAGE (creates better neural connections):
- BAD: "PostgreSQL" (flat, no context — creates orphan neuron with zero connections)
- GOOD: "Chose PostgreSQL over MongoDB because ACID needed for payments" (causal, linked)
- GOOD: "After upgrading to v3, the auth middleware broke because of new cookie format" (temporal + causal)
- GOOD: "Redis replaced Memcached for session store — 3x faster for our read pattern" (comparative + decisional)

REMEMBER PATTERNS (mix these for maximum brain connectivity):
- Causal: "X caused Y", "Because of A, B happened", "Root cause was X which led to Y"
- Temporal: "After A, then B", "Before doing X, must complete Y"
- Relational: "X depends on Y", "X replaced Y", "X connects to Y through Z"
- Decisional: "Chose X over Y because Z", "Rejected X due to Y"
- Comparative: "X is faster than Y for Z scenario"

PRIORITY SCALE: 0-3=routine notes, 5=normal, 7-8=important decisions, 9-10=critical errors/security.
TAGS: Always include project name + topic. Keep lowercase: "react", "auth", "bug-fix".

BRAIN GROWTH TIPS:
- Store 3-5 memories per task (not just 1). A bug fix has: root cause, fix, insight, prevention.
- Use diverse types: don't just store "fact" — use decision, insight, error, workflow, preference.
- Link memories: mention the same entities across memories to build neural connections.
- High-priority memories (7+) get boosted in retrieval — use them for things you'll need again.
- After storing, call pugbrain_recall on the topic to reinforce the new connections.

MEMORY CORRECTION:
- Wrong type? → pugbrain_edit(memory_id="...", type="correct_type")
- Wrong content? → pugbrain_edit(memory_id="...", content="corrected text")
- Outdated? → pugbrain_forget(memory_id="...", reason="outdated")
- Sensitive/garbage? → pugbrain_forget(memory_id="...", hard=true)

BRAIN HEALTH: Run pugbrain_health() weekly. Fix the highest top_penalty first:
- Consolidation 0% → pug consolidate (normal for new brains, run after 1 week)
- Orphans > 20% → pug consolidate --strategy prune
- Low activation → Recall 5+ different topics with pugbrain_recall
- Low connectivity → Store with context: "X because Y", "after A then B"
- Low diversity → Use varied memory types (not just facts)

CONNECTION TRACING: Use pugbrain_explain(entity_a, entity_b) to trace paths between concepts.

COGNITIVE REASONING: Use pugbrain_hypothesize to form hypotheses, pugbrain_evidence to update them,
pugbrain_predict for falsifiable predictions, pugbrain_verify to check outcomes.
Use pugbrain_schema(action="evolve") when a hypothesis needs updating.
Use pugbrain_gaps(action="detect") when you notice the brain doesn't know something.

ALL 39 TOOLS: pugbrain_remember, pugbrain_recall, pugbrain_context, pugbrain_todo, pugbrain_auto, pugbrain_suggest,
pugbrain_session, pugbrain_eternal, pugbrain_recap, pugbrain_stats, pugbrain_health, pugbrain_evolution, pugbrain_habits,
pugbrain_version, pugbrain_transplant, pugbrain_conflicts, pugbrain_alerts, pugbrain_index, pugbrain_train, pugbrain_train_db,
pugbrain_pin, pugbrain_review, pugbrain_narrative, pugbrain_import, pugbrain_explain, pugbrain_hypothesize, pugbrain_evidence,
pugbrain_predict, pugbrain_verify, pugbrain_cognitive, pugbrain_gaps, pugbrain_schema, pugbrain_edit, pugbrain_forget,
pugbrain_sync, pugbrain_sync_status, pugbrain_sync_config, pugbrain_telegram_backup, pugbrain_drift.

NEVER skip remembering after completing a feature, fixing a bug, or making a decision.
Each session starts fresh — without explicit saves, ALL discoveries are lost forever.\
"""


def get_mcp_instructions() -> str:
    """Get concise behavioral instructions for MCP InitializeResult.

    These instructions are injected into the agent's system context
    automatically by MCP clients that support the `instructions` field.
    Keep under ~200 words — behavioral directives, not documentation.

    Returns:
        Concise instruction string for proactive memory usage.
    """
    return MCP_INSTRUCTIONS


def get_prompt_for_mcp() -> dict[str, str]:
    """Get prompt formatted for MCP resources."""
    return {
        "uri": "pugbrain://prompt/system",
        "name": "PugBrain System Prompt",
        "description": "Instructions for AI assistants on using PugBrain",
        "mimeType": "text/plain",
        "text": SYSTEM_PROMPT,
    }
