# CLI Guide

Guide to using the PugBrain CLI with examples and common workflows.

!!! info "See also"
    For a complete auto-generated reference of all 66 commands, see the [CLI Reference](cli-reference.md).
    For MCP tool usage in Claude Code, see the [MCP Tools Reference](../api/mcp-tools.md).

## Core Commands

### pugbrain remember

Store a memory in the brain.

```bash
pugbrain remember "content" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--tag` | `-t` | Tags for the memory (repeatable) |
| `--type` | `-T` | Memory type (auto-detected if not specified) |
| `--priority` | `-p` | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires` | `-e` | Days until expiry |
| `--project` | `-P` | Associate with project |
| `--shared` | `-S` | Use shared/remote storage |
| `--force` | `-f` | Store even if sensitive content detected |
| `--redact` | `-r` | Auto-redact sensitive content |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
pugbrain remember "Fixed auth bug with null check"
pugbrain remember "We decided to use PostgreSQL" --type decision
pugbrain remember "Refactor auth module" --type todo --priority 7
pugbrain remember "Meeting notes" --expires 7 --tag meeting
pugbrain remember "Team knowledge" --shared
```

### pugbrain recall

Query memories using spreading activation.

```bash
pugbrain recall "query" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--depth` | `-d` | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--max-tokens` | `-m` | Max tokens in response (default: 500) |
| `--min-confidence` | `-c` | Minimum confidence threshold |
| `--shared` | `-S` | Use shared/remote storage |
| `--show-age` | `-a` | Show memory ages (default: true) |
| `--show-routing` | `-R` | Show query routing info |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
pugbrain recall "auth bug fix"
pugbrain recall "meetings with Alice" --depth 2
pugbrain recall "Why did the build fail?" --show-routing
pugbrain recall "team decisions" --shared --min-confidence 0.5
```

### pugbrain todo

Quick shortcut for TODO items.

```bash
pugbrain todo "task" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--priority` | `-p` | Priority 0-10 (default: 5) |
| `--project` | `-P` | Associate with project |
| `--expires` | `-e` | Days until expiry (default: 30) |
| `--tag` | `-t` | Tags (repeatable) |
| `--json` | `-j` | Output as JSON |

### pugbrain context

Get recent memories for context injection.

```bash
pugbrain context [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-l` | Number of recent memories (default: 10) |
| `--fresh-only` | | Only include memories < 30 days old |
| `--json` | `-j` | Output as JSON |

### pugbrain list

List memories with filters.

```bash
pugbrain list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-T` | Filter by memory type |
| `--min-priority` | `-p` | Minimum priority |
| `--project` | `-P` | Filter by project |
| `--expired` | `-e` | Show only expired memories |
| `--include-expired` | | Include expired in results |
| `--limit` | `-l` | Maximum results (default: 20) |
| `--json` | `-j` | Output as JSON |

### pugbrain stats

Show brain statistics.

```bash
pugbrain stats [--json]
```

### pugbrain check

Check content for sensitive information.

```bash
pugbrain check "content" [--json]
```

### pugbrain cleanup

Clean expired memories.

```bash
pugbrain cleanup [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--expired` | `-e` | Only clean expired (default: true) |
| `--type` | `-T` | Only clean specific type |
| `--dry-run` | `-n` | Preview without deleting |
| `--force` | `-f` | Skip confirmation |

### pugbrain consolidate

Consolidate brain memories: prune weak links, merge overlapping fibers,
advance episodic memories to semantic stage, and more.

```bash
pugbrain consolidate [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--brain` | `-b` | Brain to consolidate (default: current) |
| `--strategy` | `-s` | Strategy to run (default: `all`) |
| `--dry-run` | `-n` | Preview without applying changes |
| `--prune-threshold` | | Synapse weight threshold for pruning (default: 0.05) |
| `--merge-overlap` | | Jaccard overlap threshold for merging (default: 0.5) |
| `--min-inactive-days` | | Minimum inactive days before pruning (default: 7.0) |

**Valid strategies:**

| Strategy | Description |
|----------|-------------|
| `prune` | Remove weak synapses and orphaned neurons |
| `merge` | Combine overlapping fibers |
| `summarize` | Create concept neurons for topic clusters |
| `mature` | Advance episodic memories to semantic stage |
| `infer` | Add inferred synapses from co-activation patterns |
| `enrich` | Enrich neurons with extracted metadata |
| `dream` | Generate synthetic bridging memories |
| `learn_habits` | Extract recurring workflow patterns |
| `dedup` | Merge near-duplicate memories |
| `semantic_link` | Add cross-domain semantic connections |
| `compress` | Compress old low-activation fibers |
| `all` | Run all strategies in dependency order (default) |

> **Note:** `mature` is a fully supported strategy. It advances episodic memories
> to the semantic stage, which improves recall quality. `pugbrain health` may recommend
> running it when the consolidation ratio is low.

**Examples:**

```bash
pugbrain consolidate                          # Run all strategies
pugbrain consolidate --strategy prune         # Only prune weak links
pugbrain consolidate -s mature                # Advance episodic memories
pugbrain consolidate --dry-run                # Preview without changes
pugbrain consolidate -s merge --merge-overlap 0.3
```

> **Tip:** Always use `--strategy <name>` (named flag). Positional syntax
> (`pugbrain consolidate prune`) is not supported and will produce a helpful error.

### pugbrain decay

Apply memory decay (Ebbinghaus forgetting curve).

```bash
pugbrain decay [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--brain` | Brain name (default: current) |
| `--dry-run` | Preview without applying |
| `--prune-threshold` | Threshold for pruning (default: 0.01) |

---

## Brain Commands

### pugbrain brain list

List all brains.

```bash
pugbrain brain list [--json]
```

### pugbrain brain create

Create a new brain.

```bash
pugbrain brain create NAME [--use/--no-use]
```

### pugbrain brain use

Switch to a brain.

```bash
pugbrain brain use NAME
```

### pugbrain brain export

Export brain to file.

```bash
pugbrain brain export [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path |
| `--name` | `-n` | Brain name (default: current) |
| `--exclude-sensitive` | `-s` | Exclude sensitive content |

### pugbrain brain import

Import brain from file.

```bash
pugbrain brain import FILE [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Name for imported brain |
| `--use` | `-u` | Switch to imported brain |
| `--merge` | | Merge with existing brain |
| `--scan` | | Scan for sensitive content |

### pugbrain brain delete

Delete a brain.

```bash
pugbrain brain delete NAME [--force]
```

### pugbrain brain health

Check brain health.

```bash
pugbrain brain health [--name NAME] [--json]
```

---

## Project Commands

### pugbrain project create

Create a project.

```bash
pugbrain project create NAME [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--description` | `-d` | Project description |
| `--duration` | `-D` | Duration in days |
| `--tag` | `-t` | Tags (repeatable) |
| `--priority` | `-p` | Priority (default: 1.0) |

### pugbrain project list

List projects.

```bash
pugbrain project list [--active] [--json]
```

### pugbrain project show

Show project details.

```bash
pugbrain project show NAME [--json]
```

### pugbrain project delete

Delete a project.

```bash
pugbrain project delete NAME [--force]
```

### pugbrain project extend

Extend project deadline.

```bash
pugbrain project extend NAME DAYS [--json]
```

---

## Shared Mode Commands

### pugbrain shared enable

Enable shared storage mode.

```bash
pugbrain shared enable URL [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--api-key` | `-k` | API key for authentication |
| `--timeout` | `-t` | Request timeout in seconds |

### pugbrain shared disable

Disable shared mode.

```bash
pugbrain shared disable
```

### pugbrain shared status

Show shared mode status.

```bash
pugbrain shared status [--json]
```

### pugbrain shared test

Test server connection.

```bash
pugbrain shared test
```

### pugbrain shared sync

Sync with server.

```bash
pugbrain shared sync [--direction push|pull|both] [--json]
```

---

## Telegram Commands

### pugbrain telegram status

Show Telegram integration configuration status.

```bash
pugbrain telegram status [--json]
```

Shows: bot token configured (yes/no), bot name/username, chat IDs, backup-on-consolidation flag.

### pugbrain telegram test

Send a test message to all configured Telegram chats.

```bash
pugbrain telegram test [--json]
```

Verifies bot token and chat IDs are working. Sends a "PugBrain test" message.

### pugbrain telegram backup

Send brain .db file as backup to all configured Telegram chats.

```bash
pugbrain telegram backup [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--brain` | `-b` | Brain name (default: current) |
| `--json` | `-j` | Output as JSON |

**Setup:**

1. Set bot token: `export NMEM_TELEGRAM_BOT_TOKEN="your-bot-token"`
2. Add chat IDs to `~/.pugbrain/config.toml`:

```toml
[telegram]
enabled = true
chat_ids = ["123456789"]
```

**Examples:**

```bash
pugbrain telegram status               # Check config
pugbrain telegram test                  # Verify bot works
pugbrain telegram backup                # Backup current brain
pugbrain telegram backup --brain work   # Backup specific brain
```

---

## Server Commands

### pugbrain serve

Start the FastAPI server.

```bash
pugbrain serve [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--host` | Host to bind (default: 127.0.0.1) |
| `--port` | Port to bind (default: 8000) |
| `--reload` | Enable auto-reload for development |

### pugbrain mcp

Start MCP server for Claude integration.

```bash
pugbrain mcp
```

### pugbrain prompt

Show MCP system prompt.

```bash
pugbrain prompt [--compact] [--json]
```

### pugbrain mcp-config

Show MCP configuration.

```bash
pugbrain mcp-config
```

### pugbrain install-skills

Install bundled agent skills to `~/.claude/skills/`.

```bash
pugbrain install-skills [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Overwrite existing skills with latest version |
| `--list` | `-l` | List available skills without installing |

**Examples:**

```bash
pugbrain install-skills            # Install all skills
pugbrain install-skills --force    # Overwrite with latest
pugbrain install-skills --list     # Show available skills
```

---

## Memory Types

| Type | Description | Default Expiry |
|------|-------------|----------------|
| `fact` | Objective information | Never |
| `decision` | Choices made | Never |
| `preference` | User preferences | Never |
| `todo` | Action items | 30 days |
| `insight` | Learned patterns | Never |
| `context` | Situational info | 7 days |
| `instruction` | User guidelines | Never |
| `error` | Error patterns | Never |
| `workflow` | Process patterns | Never |
| `reference` | External references | Never |

## Depth Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Instant | Direct recall (who, what, where) |
| 1 | Context | Before/after context (2-3 hops) |
| 2 | Habit | Cross-time patterns |
| 3 | Deep | Full causal/emotional analysis |
