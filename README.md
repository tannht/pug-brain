# PugBrain: Reflex-based Neural Memory

> **Reflex-based memory system for AI agents — retrieval through activation, not search.**

PugBrain is a hybrid neural-vector memory system for long-term agent continuity. It combines graph relationships with vector embeddings for reflexive recall.

PugBrain stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works. Instead of searching a database, memories surface through associative recall — activating related concepts until the relevant memory emerges.

**45 MCP tools** · **14 memory types** · **24 synapse types** · **Schema v26** · **3778+ tests** · **Cognitive reasoning layer**

## Why Not RAG / Vector Search?

| Aspect | RAG / Vector Search | PugBrain |
|--------|---------------------|----------|
| **Model** | Search engine | Human brain |
| **LLM/Embedding** | Required (embedding API calls) | **Optional** — core recall is pure algorithmic graph traversal |
| **Query** | "Find similar text" | "Recall through association" |
| **Structure** | Flat chunks + embeddings | Neural graph + synapses |
| **Relationships** | None (just similarity) | Explicit: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, etc. |
| **Temporal** | Timestamp filter | Time as first-class neurons |
| **Multi-hop** | Multiple queries needed | Natural graph traversal |
| **Lifecycle** | Static | Decay, reinforcement, consolidation |
| **API Cost** | ~$0.02/1K queries | **$0.00** — fully offline (optional embeddings available) |

**Example: "Why did Tuesday's outage happen?"**

- **RAG**: Returns "JWT caused outage" (missing *why* we used JWT)
- **PugBrain**: Traces `outage ← CAUSED_BY ← JWT ← SUGGESTED_BY ← Alice` → full causal chain

---

## Installation

```bash
pip install pug-brain
```

## Setup

### Claude Code (Plugin)

```bash
/plugin marketplace add tannht/pug-brain
/plugin install pug-brain@pug-brain-marketplace
```

### Cursor / Windsurf / Other MCP Clients

```bash
pip install pug-brain
```

Add to your editor's MCP config:

```json
{
  "mcpServers": {
    "pugbrain": {
      "command": "pug-mcp"
    }
  }
}
```

Auto-initializes on first use — no `pugbrain init` needed.

### OpenClaw

```bash
pip install pug-brain && npm install -g pugbrain
```

```json
{ "plugins": { "slots": { "memory": "pugbrain" } } }
```

See the [full setup guide](docs/guides/openclaw-plugin.md).

## Installation Options

```bash
pip install pug-brain[server]           # FastAPI server + dashboard
pip install pug-brain[extract]          # PDF/DOCX/PPTX/HTML/XLSX/CSV extraction
pip install pug-brain[embeddings]       # Local embedding (cross-language recall)
pip install pug-brain[embeddings-openai] # OpenAI embeddings
pip install pug-brain[embeddings-gemini] # Google Gemini embeddings
pip install pug-brain[nlp-vi]           # Vietnamese NLP
pip install pug-brain[neo4j]            # Neo4j backend
pip install pug-brain[falkordb]         # FalkorDB backend
pip install pug-brain[encryption]       # Encrypted storage
pip install pug-brain[all]              # All features
```

### Cross-Language Recall (Optional)

```toml
# ~/.pugbrain/config.toml
[embedding]
enabled = true
provider = "auto"    # Ollama -> sentence-transformers -> Gemini -> OpenAI
```

## How It Works

```
Query: "What did Alice suggest?"
  1. Decompose Query   -> time hints, entities, intent
  2. Find Anchors      -> "Alice" neuron
  3. Spread Activation -> activate connected neurons
  4. Find Intersection -> high-activation subgraph
  5. Extract Context   -> "Alice suggested rate limiting"
```

| Concept | Description |
|---------|-------------|
| **Neuron** | Memory unit (concept, entity, action, time, state, spatial, sensory, intent) |
| **Synapse** | Weighted, typed connection (`CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, ...) |
| **Fiber** | Ordered neuron sequence forming a coherent memory trace |
| **Spreading activation** | Signal propagates from anchors through synapses, decaying with distance |
| **Decay** | Ebbinghaus forgetting curve — memories lose activation over time |
| **Consolidation** | Prune weak synapses, merge overlapping fibers, summarize clusters |

## CLI

```bash
# Store (type auto-detected)
pugbrain remember "Fixed auth bug in login.py:42"
pugbrain remember "We decided to use PostgreSQL" --type decision
pugbrain todo "Review PR #123" --priority 7

# Recall
pugbrain recall "auth bug"
pugbrain recall "database decision" --depth 2

# Shortcuts
pugbrain a "quick note"          # remember
pugbrain q "auth"                # recall
pugbrain last 5                  # recent memories
pugbrain today                   # today's memories

# Brain management
pugbrain brain list | create | use | health | export | import

# Maintenance
pugbrain decay                   # Forgetting curve
pugbrain consolidate             # Prune, merge, summarize
pugbrain index src/              # Index codebase

# Server & backup
pugbrain serve                   # FastAPI + dashboard
pugbrain telegram backup         # Telegram backup
```

## Python API

```python
import asyncio
from neural_memory import Brain
from neural_memory.storage import InMemoryStorage
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import ReflexPipeline

async def main():
    storage = InMemoryStorage()
    brain = Brain.create("my_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Decided to use FastAPI for backend")

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("What did we decide about backend?")
    print(result.context)

asyncio.run(main())
```

## MCP Tools (45)

Once configured, these 45 tools are available to your AI assistant:

**Core Memory:**

| Tool | Description |
|------|-------------|
| `pugbrain_remember` | Store memory (auto-detects type: fact, decision, insight, error, etc.) |
| `pugbrain_recall` | Query with spreading activation (4 depth levels) |
| `pugbrain_context` | Recent memories as session context |
| `pugbrain_todo` | Quick TODO with 30-day expiry |
| `pugbrain_auto` | Auto-capture memories from conversation text |
| `pugbrain_suggest` | Autocomplete from brain neurons |
| `pugbrain_edit` | Edit type, content, or priority (preserves connections) |
| `pugbrain_forget` | Soft/hard delete |

### Workflow (5)

| Tool | Description |
|------|-------------|
| `pugbrain_session` | Track task, feature, progress |
| `pugbrain_eternal` | Save project context, decisions, instructions |
| `pugbrain_recap` | Load saved context at session start |
| `pugbrain_stats` | Brain statistics and metrics |
| `pugbrain_habits` | Workflow habit suggestions |

### Knowledge Base (4)

| Tool | Description |
|------|-------------|
| `pugbrain_train` | Train from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD) |
| `pugbrain_train_db` | Train from database schema |
| `pugbrain_index` | Index codebase for code-aware recall |
| `pugbrain_pin` | Pin/unpin memories (pinned = permanent, skip decay) |

### Cognitive Reasoning (8)

| Tool | Description |
|------|-------------|
| `pugbrain_hypothesize` | Hypotheses with Bayesian confidence tracking |
| `pugbrain_evidence` | Submit evidence for/against — auto-updates confidence |
| `pugbrain_predict` | Falsifiable predictions with deadlines |
| `pugbrain_verify` | Verify predictions — propagates to linked hypotheses |
| `pugbrain_cognitive` | Hot index of active hypotheses + predictions |
| `pugbrain_gaps` | Detect and track knowledge gaps |
| `pugbrain_schema` | Evolve hypotheses with SUPERSEDES version chains |
| `pugbrain_explain` | Trace shortest path between two concepts |

### Advanced (12)

| Tool | Description |
|------|-------------|
| `pugbrain_health` | Purity score, grade (A-F), top penalties with fix actions |
| `pugbrain_review` | Spaced repetition (Leitner box system) |
| `pugbrain_conflicts` | List, resolve, or pre-check memory conflicts |
| `pugbrain_narrative` | Generate timeline, topic, or causal chain narratives |
| `pugbrain_alerts` | Health alerts: list or acknowledge |
| `pugbrain_version` | Snapshot, list, rollback, diff |
| `pugbrain_transplant` | Transplant memories between brains |
| `pugbrain_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `pugbrain_sync` | Multi-device sync (push/pull/full) |
| `pugbrain_sync_status` | Sync status and pending changes |
| `pugbrain_sync_config` | Configure sync settings |
| `pugbrain_telegram_backup` | Send brain backup to Telegram |

### Provenance & Show (3+)

| Tool | Description |
|------|-------------|
| `pugbrain_show` | Full verbatim content + metadata for a memory by ID |
| `pugbrain_provenance` | Trace/verify/approve memory audit trail |
| `pugbrain_source` | Register and manage external sources |

## Memory Types

`fact` · `decision` · `preference` · `todo` · `insight` · `context` · `instruction` · `error` · `workflow` · `reference`

```bash
pugbrain remember "We chose X over Y" --type decision
pugbrain remember "Always validate input" --type instruction
pugbrain todo "Review PR" --priority 7 --expires 30
```

## Dashboard (8 pages)

```bash
pugbrain serve    # http://localhost:8000/dashboard
```

| Page | Description |
|------|-------------|
| **Overview** | KPI cards + brain table with switch/delete |
| **Health** | Radar chart + warnings + recommendations |
| **Graph** | Sigma.js WebGL neural graph, ForceAtlas2, node inspector |
| **Timeline** | Chronological memory feed with type badges |
| **Evolution** | Maturity, plasticity, stage distribution charts |
| **Diagrams** | Interactive fiber mindmap (dagre layout, zoom/pan) |
| **Settings** | Brain files, Telegram backup config |
| **Neurodungeon** | Roguelike dungeon crawler powered by brain data |

Light/Dark/System theme toggle.

## Cognitive Reasoning

```bash
# Hypothesis -> Evidence -> Prediction -> Verification cycle
pugbrain_hypothesize(action="create", content="Redis causes latency", confidence=0.6)
pugbrain_evidence(hypothesis_id="h-1", type="for", content="Redis at 200ms")
pugbrain_predict(action="create", content="Valkey will fix it", hypothesis_id="h-1")
pugbrain_verify(prediction_id="p-1", outcome="correct")

# Schema evolution
pugbrain_schema(action="evolve", hypothesis_id="h-1",
    content="Network was root cause", reason="New evidence")

# Knowledge gaps
pugbrain_gaps(action="detect", topic="Why latency at 3am?")
```

Auto-resolution: confidence >= 0.9 + 3 supporting evidence -> confirmed. <= 0.1 + 3 against -> refuted.

## Brain Health

7 components: Connectivity (25%), Diversity (20%), Freshness (15%), Consolidation (15%), Orphan Rate (10%), Activation (10%), Recall Confidence (5%).

Reports include **`top_penalties`** with exact fix actions. See the [Brain Health Guide](docs/guides/brain-health.md).

## Server API

```bash
pip install pug-brain[server]
pugbrain serve                    # localhost:8000
```

Returns the path with evidence: `Redis → USED_BY → session-store → CAUSED_BY → auth outage`. Use this to debug recall, verify brain connections, or discover unexpected relationships between concepts.

### Cognitive Reasoning

Hypothesize, predict, verify, and evolve beliefs — the brain reasons about what it knows:

```bash
# Create a hypothesis with initial confidence
pugbrain_hypothesize(action="create", content="Redis is causing the latency spike", confidence=0.6)

# Submit evidence
pugbrain_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency at 200ms")
pugbrain_evidence(hypothesis_id="h-1", evidence_type="against", content="Network latency was 500ms")

# Make a falsifiable prediction
pugbrain_predict(action="create", content="Switching to Valkey will fix latency",
             hypothesis_id="h-1", deadline="2026-04-01")

# Verify prediction outcome — propagates to linked hypothesis
pugbrain_verify(prediction_id="p-1", outcome="correct")

# Evolve hypothesis when understanding changes (creates SUPERSEDES chain)
pugbrain_schema(action="evolve", hypothesis_id="h-1",
            content="Network config was root cause, not Redis",
            reason="New evidence from network team")

# Track what the brain doesn't know
pugbrain_gaps(action="detect", topic="Why does latency spike at 3am?", source="recall_miss")

# View cognitive dashboard
pugbrain_cognitive(action="summary")    # Hot index of active hypotheses + predictions
pugbrain_schema(action="history", hypothesis_id="h-2")  # Version evolution chain
```

Auto-resolution: hypotheses with confidence ≥0.9 + 3 supporting evidence → auto-confirmed. Confidence ≤0.1 + 3 against → auto-refuted. Calibration score tracks prediction accuracy.

### Brain Versioning

```bash
pugbrain_version(action="create", name="v1-stable")  # Snapshot
pugbrain_version(action="list")                       # List versions
pugbrain_version(action="rollback", version_id="...")  # Restore
pugbrain_version(action="diff", from_version="...", to_version="...")
```

### Web Dashboard

```bash
pugbrain serve                         # Start server on localhost:8000
# Open http://localhost:8000/dashboard  # React dashboard (7 pages)
# Open http://localhost:8000/docs       # API docs (Swagger)
```

Pages:
- **Overview** — KPI cards (neurons, synapses, fibers, brains) + brain table with click-to-switch and delete
- **Health** — Radar chart + health warnings + recommendations
- **Graph** — Sigma.js WebGL neural graph with ForceAtlas2 layout, color-coded by type, node inspector
- **Timeline** — Chronological memory feed with type badges
- **Evolution** — Brain maturity, plasticity, stage distribution charts
- **Mindmap** — ReactFlow interactive fiber mindmap (dagre tree, zoom/pan, MiniMap)
- **Settings** — Brain files, Telegram backup config

Light/Dark/System theme toggle with warm cream light mode.

### Telegram Backup

Send brain `.db` files to Telegram for offsite backup:

```bash
# Setup: set env var + config
export PUGBRAIN_TELEGRAM_BOT_TOKEN="your-bot-token"
# Add to config.toml:
# [telegram]
# enabled = true
# chat_ids = ["123456789"]

# CLI
pugbrain telegram status              # Check config
pugbrain telegram test                # Send test message
pugbrain telegram backup              # Send brain backup
pugbrain telegram backup --brain work # Specific brain

# MCP tool
pugbrain_telegram_backup(brain_name="work")
```

### Cloud Sync (Multi-Device)

Sync memories across all your devices with one command:

```python
# 1. Get your API key (one-time)
pugbrain_sync_config(action="setup")       # Shows registration steps

# 2. Connect
pugbrain_sync_config(action="set",
    hub_url="https://pug-brain-sync-hub.vietnam11399.workers.dev",
    api_key="nmk_YOUR_KEY")

# 3. Sync
pugbrain_sync(action="seed")              # Prepare existing memories
pugbrain_sync(action="push")              # Push to cloud
pugbrain_sync(action="pull")              # Pull on another device
pugbrain_sync(action="full")              # Bidirectional sync
pugbrain_sync_status()                    # Check sync status & devices
```

See the full [Cloud Sync Guide](https://nhadaututtheky.github.io/pug-brain/guides/cloud-sync/) for key management, conflict resolution, and troubleshooting.

### External Memory Import

Import from existing memory systems:

```bash
# ChromaDB
pugbrain import backup.json --source chromadb

# Via MCP tool
pugbrain_import(source="mem0")           # Uses MEM0_API_KEY env var
pugbrain_import(source="chromadb", connection="/path/to/chroma")
pugbrain_import(source="cognee")         # Uses COGNEE_API_KEY env var
pugbrain_import(source="graphiti", connection="bolt://localhost:7687")
pugbrain_import(source="llamaindex", connection="/path/to/index")
```

### Safety & Security

```bash
# Sensitive content detection
pugbrain check "API_KEY=sk-xxx"

# Auto-redact before storing
pugbrain remember "Config: API_KEY=sk-xxx" --redact

# Safe export (exclude sensitive neurons)
pugbrain brain export --exclude-sensitive -o safe.json

# Health check (freshness + sensitive scan)
pugbrain brain health
```

- Content length validation (100KB limit)
- ReDoS protection (text truncation before regex)
- Spreading activation queue cap (prevents memory exhaustion)
- API keys read from environment variables, never from tool parameters
- `max_tokens` clamped to 10,000

### Server Mode

```bash
pip install pug-brain[server]
pugbrain serve                    # localhost:8000
pugbrain serve -p 9000            # Custom port
pugbrain serve --host 0.0.0.0    # Expose to network
```

API endpoints:
```
POST /memory/encode     - Store memory
POST /memory/query      - Query memories
POST /brain/create      - Create brain
GET  /brain/{id}/export - Export brain
WS   /sync/ws           - Real-time sync (local server)
POST /v1/hub/sync       - Cloud sync (push/pull/full)
POST /v1/hub/register   - Register device for sync
GET  /v1/hub/status     - Hub sync status
GET  /dashboard         - Web dashboard
GET  /docs              - Swagger API docs
```

## Safety & Security

- Parameterized SQL only — no string interpolation
- Content length validation (100KB limit)
- ReDoS protection (text truncation before regex)
- Spreading activation queue cap (prevents memory exhaustion)
- API keys from environment variables only
- `max_tokens` clamped to 10,000
- Sensitive content auto-detection and redaction
- Default bind to `127.0.0.1`, CORS restricted to localhost

## VS Code Extension

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory).

Memory tree view, interactive graph explorer, CodeLens memory counts, encode from selections, real-time WebSocket sync.

## Development

```bash
git clone https://github.com/tannht/pug-brain
cd pug-brain
pip install -e ".[dev]"
pytest tests/ -v              # 3,700+ tests
ruff check src/ tests/        # Lint
ruff format src/ tests/       # Format
```

## Documentation

- [Complete Guide](docs/index.md)
- [Integration Guide](docs/guides/integration.md)
- [Safety & Limitations](docs/guides/safety.md)
- [Architecture](docs/architecture/overview.md)
- [Embedding Setup](docs/guides/embedding-setup.md)

## Support

**Solana:** `5XVY6dZDeyuZJy6Co9KeLDxY5RZ6EwCpjsUVkacMz7HF`

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
