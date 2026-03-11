# PugBrain: Reflex-based Neural Memory

> **Reflex-based memory system for AI agents — retrieval through activation, not search.**

PugBrain is a hybrid neural-vector memory system for long-term agent continuity. It combines graph relationships with vector embeddings for reflexive recall.

**45 MCP tools** · **14 memory types** · **24 synapse types** · **Schema v22** · **3,700+ tests** · **Cognitive reasoning layer**

```bash
pip install neural-memory
```

## Setup

### Claude Code (Plugin)

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

### Cursor / Windsurf / Other MCP Clients

```bash
pip install neural-memory
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
pip install neural-memory && npm install -g pugbrain
```

```json
{ "plugins": { "slots": { "memory": "pugbrain" } } }
```

See the [full setup guide](docs/guides/openclaw-plugin.md).

## Installation Options

```bash
pip install neural-memory[server]           # FastAPI server + dashboard
pip install neural-memory[extract]          # PDF/DOCX/PPTX/HTML/XLSX/CSV extraction
pip install neural-memory[embeddings]       # Local embedding (cross-language recall)
pip install neural-memory[embeddings-openai] # OpenAI embeddings
pip install neural-memory[embeddings-gemini] # Google Gemini embeddings
pip install neural-memory[nlp-vi]           # Vietnamese NLP
pip install neural-memory[neo4j]            # Neo4j backend
pip install neural-memory[falkordb]         # FalkorDB backend
pip install neural-memory[encryption]       # Encrypted storage
pip install neural-memory[all]              # All features
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

### Core Memory (8)

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
pip install neural-memory[server]
pugbrain serve                    # localhost:8000
```

```
POST /memory/encode     - Store memory
POST /memory/query      - Query memories
POST /brain/create      - Create brain
GET  /brain/{id}/export - Export brain
WS   /sync/ws           - Real-time sync
POST /hub/sync          - Multi-device sync
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
git clone https://github.com/nhadaututtheky/neural-memory
cd neural-memory
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
