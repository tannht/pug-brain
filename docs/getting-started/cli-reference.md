# CLI Reference

Complete reference for the `pugbrain` command-line interface.
**66 commands** available.

!!! tip
    Run `pugbrain --help` or `pugbrain <command> --help` for the latest usage info.

## Table of Contents

- [Memory Operations](#memory)
  - [`pugbrain remember`](#pugbrain-remember)
  - [`pugbrain recall`](#pugbrain-recall)
  - [`pugbrain context`](#pugbrain-context)
  - [`pugbrain todo`](#pugbrain-todo)
  - [`pugbrain q`](#pugbrain-q)
  - [`pugbrain a`](#pugbrain-a)
  - [`pugbrain last`](#pugbrain-last)
  - [`pugbrain today`](#pugbrain-today)
- [Brain Management](#brain)
  - [`pugbrain brain list`](#pugbrain-brain-list)
  - [`pugbrain brain use`](#pugbrain-brain-use)
  - [`pugbrain brain create`](#pugbrain-brain-create)
  - [`pugbrain brain export`](#pugbrain-brain-export)
  - [`pugbrain brain import`](#pugbrain-brain-import)
  - [`pugbrain brain delete`](#pugbrain-brain-delete)
  - [`pugbrain brain health`](#pugbrain-brain-health)
  - [`pugbrain brain transplant`](#pugbrain-brain-transplant)
- [Information & Diagnostics](#info)
  - [`pugbrain stats`](#pugbrain-stats)
  - [`pugbrain status`](#pugbrain-status)
  - [`pugbrain health`](#pugbrain-health)
  - [`pugbrain check`](#pugbrain-check)
  - [`pugbrain doctor`](#pugbrain-doctor)
  - [`pugbrain dashboard`](#pugbrain-dashboard)
  - [`pugbrain ui`](#pugbrain-ui)
  - [`pugbrain graph`](#pugbrain-graph)
- [Training & Import/Export](#training)
  - [`pugbrain train`](#pugbrain-train)
  - [`pugbrain index`](#pugbrain-index)
  - [`pugbrain import`](#pugbrain-import)
  - [`pugbrain export`](#pugbrain-export)
- [Configuration & Setup](#config)
  - [`pugbrain init`](#pugbrain-init)
  - [`pugbrain setup`](#pugbrain-setup)
  - [`pugbrain mcp-config`](#pugbrain-mcp-config)
  - [`pugbrain prompt`](#pugbrain-prompt)
  - [`pugbrain hooks`](#pugbrain-hooks)
  - [`pugbrain config preset`](#pugbrain-config-preset)
  - [`pugbrain config tier`](#pugbrain-config-tier)
  - [`pugbrain install-skills`](#pugbrain-install-skills)
- [Server & MCP](#server)
  - [`pugbrain serve`](#pugbrain-serve)
  - [`pugbrain mcp`](#pugbrain-mcp)
- [Maintenance](#maintenance)
  - [`pugbrain decay`](#pugbrain-decay)
  - [`pugbrain consolidate`](#pugbrain-consolidate)
  - [`pugbrain cleanup`](#pugbrain-cleanup)
  - [`pugbrain flush`](#pugbrain-flush)
- [Project Management](#project)
  - [`pugbrain project create`](#pugbrain-project-create)
  - [`pugbrain project list`](#pugbrain-project-list)
  - [`pugbrain project show`](#pugbrain-project-show)
  - [`pugbrain project delete`](#pugbrain-project-delete)
  - [`pugbrain project extend`](#pugbrain-project-extend)
- [Advanced Features](#advanced)
  - [`pugbrain shared enable`](#pugbrain-shared-enable)
  - [`pugbrain shared disable`](#pugbrain-shared-disable)
  - [`pugbrain shared status`](#pugbrain-shared-status)
  - [`pugbrain shared test`](#pugbrain-shared-test)
  - [`pugbrain shared sync`](#pugbrain-shared-sync)
  - [`pugbrain habits list`](#pugbrain-habits-list)
  - [`pugbrain habits show`](#pugbrain-habits-show)
  - [`pugbrain habits clear`](#pugbrain-habits-clear)
  - [`pugbrain habits status`](#pugbrain-habits-status)
  - [`pugbrain version create`](#pugbrain-version-create)
  - [`pugbrain version list`](#pugbrain-version-list)
  - [`pugbrain version rollback`](#pugbrain-version-rollback)
  - [`pugbrain version diff`](#pugbrain-version-diff)
  - [`pugbrain telegram status`](#pugbrain-telegram-status)
  - [`pugbrain telegram test`](#pugbrain-telegram-test)
  - [`pugbrain telegram backup`](#pugbrain-telegram-backup)
  - [`pugbrain list`](#pugbrain-list)
  - [`pugbrain migrate`](#pugbrain-migrate)
  - [`pugbrain update`](#pugbrain-update)

---

## Memory Operations {#memory}

### `pugbrain remember`

Store a new memory (type auto-detected if not specified).

```
pugbrain remember [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | No | `` | (positional argument) |
| `--tag / -t` | text | No | — | Tags for the memory |
| `--type / -T` | text | No | — | Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detect... |
| `--priority / -p` | integer | No | — | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires / -e` | integer | No | — | Days until this memory expires |
| `--project / -P` | text | No | — | Associate with a project (by name) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--force / -f` | boolean | No | `False` | Store even if sensitive content detected |
| `--redact / -r` | boolean | No | `False` | Auto-redact sensitive content before storing |
| `--timestamp / --at` | text | No | — | ISO datetime of original event (e.g. '2026-03-02T08:00:00'). Defaults to now. |
| `--stdin` | boolean | No | `False` | Read content from stdin (safe for shell-special characters) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain recall`

Query memories with intelligent routing (query type auto-detected).

```
pugbrain recall [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `--depth / -d` | integer | No | — | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--max-tokens / -m` | integer | No | `500` | Max tokens in response |
| `--min-confidence / -c` | float | No | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--show-age / -a` | boolean | No | `True` | Show memory ages in results |
| `--show-routing / -R` | boolean | No | `False` | Show query routing info |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain context`

Get recent context (for injecting into AI conversations).

```
pugbrain context [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `10` | Number of recent memories |
| `--fresh-only` | boolean | No | `False` | Only include memories < 30 days old |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain todo`

Quick shortcut to add a TODO memory.

```
pugbrain todo [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `task` | text | Yes | — | (positional argument) |
| `--priority / -p` | integer | No | `5` | Priority 0-10 (default: 5=normal, 7=high, 10=critical) |
| `--project / -P` | text | No | — | Associate with a project |
| `--expires / -e` | integer | No | — | Days until expiry (default: 30) |
| `--tag / -t` | text | No | — | Tags for the task |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain q`

Quick recall - shortcut for 'pugbrain recall'.

```
pugbrain q [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `-d` | integer | No | — | — |

### `pugbrain a`

Quick add - shortcut for 'pugbrain remember' with auto-detect.

```
pugbrain a [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `-p` | integer | No | — | — |

### `pugbrain last`

Show last N memories - quick view of recent activity.

```
pugbrain last [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `-n` | integer | No | `5` | Number of memories to show |

### `pugbrain today`

Show today's memories.

```
pugbrain today [OPTIONS]
```

## Brain Management {#brain}

### `pugbrain brain list`

List available brains.

```
pugbrain brain list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain brain use`

Switch to a different brain.

```
pugbrain brain use [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |

### `pugbrain brain create`

Create a new brain.

```
pugbrain brain create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--use / -u` | boolean | No | `True` | Switch to the new brain after creating |

### `pugbrain brain export`

Export brain to JSON or markdown file.

```
pugbrain brain export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output / -o` | text | No | — | Output file path |
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--exclude-sensitive / -s` | boolean | No | `False` | Exclude memories with sensitive content |
| `--format / -f` | text | No | `json` | Export format: json or markdown |

### `pugbrain brain import`

Import brain from JSON file.

```
pugbrain brain import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `file` | text | Yes | — | (positional argument) |
| `--name / -n` | text | No | — | Name for imported brain |
| `--use / -u` | boolean | No | `True` | Switch to imported brain |
| `--scan` | boolean | No | `True` | Scan for sensitive content before importing |

### `pugbrain brain delete`

Delete a brain.

```
pugbrain brain delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `pugbrain brain health`

Check brain health (freshness, sensitive content).

```
pugbrain brain health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain brain transplant`

Transplant memories from another brain into the current brain.

```
pugbrain brain transplant [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | text | Yes | — | (positional argument) |
| `--tag / -t` | text | No | — | Filter by tags |
| `--type` | text | No | — | Filter by memory types |
| `--strategy / -s` | text | No | `prefer_local` | Conflict resolution strategy |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Information & Diagnostics {#info}

### `pugbrain stats`

Show brain statistics including freshness and memory type analysis.

```
pugbrain stats [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain status`

Show current brain status, recent activity, and actionable suggestions.

```
pugbrain status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain health`

Show brain health diagnostics with purity score and recommendations.

```
pugbrain health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain check`

Check content for sensitive information without storing.

```
pugbrain check [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain doctor`

Run system health diagnostics.

```
pugbrain doctor [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain dashboard`

Show a rich dashboard with brain stats and recent activity.

```
pugbrain dashboard [OPTIONS]
```

### `pugbrain ui`

Interactive memory browser with rich formatting.

```
pugbrain ui [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -t` | text | No | — | Filter by memory type |
| `--search / -s` | text | No | — | Search in memory content |
| `--limit / -n` | integer | No | `20` | Number of memories to show |

### `pugbrain graph`

Visualize neural connections as a tree graph.

```
pugbrain graph [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | No | — | (positional argument) |
| `--depth / -d` | integer | No | `2` | Traversal depth (1-3) |
| `--export / -e` | text | No | — | Export format: svg |
| `--output / -o` | text | No | — | Output file path (used with --export) |

## Training & Import/Export {#training}

### `pugbrain train`

Train a brain from documentation files (markdown).

```
pugbrain train [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--domain / -d` | text | No | `` | Domain tag (e.g., react, kubernetes) |
| `--brain / -b` | text | No | `` | Target brain name (default: current) |
| `--ext / -e` | text | No | — | File extensions (default: .md) |
| `--no-consolidate` | boolean | No | `False` | Skip ENRICH consolidation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain index`

Index a codebase into neural memory for code-aware recall.

```
pugbrain index [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--ext / -e` | text | No | — | File extensions to index (e.g. .py) |
| `--status / -s` | boolean | No | `False` | Show indexing status instead of scanning |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain import`

Import brain from JSON file.

```
pugbrain import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `input_file` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Target brain name (default: from file) |
| `--merge / -m` | boolean | No | `False` | Merge with existing brain |
| `--strategy` | text | No | `prefer_local` | Conflict resolution: prefer_local, prefer_remote, prefer_recent, prefer_stronger |

### `pugbrain export`

Export brain to JSON file for backup or sharing.

```
pugbrain export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `output` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to export (default: current) |

## Configuration & Setup {#config}

### `pugbrain init`

Set up PugBrain in one command.

```
pugbrain init [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing config |
| `--skip-mcp` | boolean | No | `False` | Skip MCP auto-configuration |
| `--skip-skills` | boolean | No | `False` | Skip skills installation |
| `--wizard / -w` | boolean | No | `False` | Interactive setup wizard |
| `--defaults` | boolean | No | `False` | Non-interactive with all defaults |

### `pugbrain setup`

Set up optional components.

```
pugbrain setup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `component` | text | No | `` | (positional argument) |

### `pugbrain mcp-config`

Generate MCP server configuration for Claude Code/Cursor.

```
pugbrain mcp-config [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--with-prompt / -p` | boolean | No | `False` | Include system prompt in config |
| `--compact / -c` | boolean | No | `False` | Use compact prompt (if --with-prompt) |

### `pugbrain prompt`

Show system prompt for AI tools.

```
pugbrain prompt [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--compact / -c` | boolean | No | `False` | Show compact version |
| `--copy` | boolean | No | `False` | Copy to clipboard (requires pyperclip) |

### `pugbrain hooks`

Install or manage git hooks for automatic memory capture.

```
pugbrain hooks [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `action` | text | No | `install` | (positional argument) |
| `--path / -p` | text | No | — | Path to git repo (default: current dir) |

### `pugbrain config preset`

Apply a configuration preset or list available presets.

```
pugbrain config preset [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--list / -l` | boolean | No | `False` | List available presets |
| `--dry-run / -n` | boolean | No | `False` | Show changes without applying |

### `pugbrain config tier`

Get or set the MCP tool tier to control token usage.

```
pugbrain config tier [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--show / -s` | boolean | No | `False` | Show current tier |

### `pugbrain install-skills`

Install PugBrain skills to ~/.claude/skills/.

```
pugbrain install-skills [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing skills with latest version |
| `--list / -l` | boolean | No | `False` | List available skills without installing |

## Server & MCP {#server}

### `pugbrain serve`

Run the PugBrain API server.

```
pugbrain serve [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--host / -h` | text | No | `127.0.0.1` | Host to bind to |
| `--port / -p` | integer | No | `8000` | Port to bind to |
| `--reload / -r` | boolean | No | `False` | Enable auto-reload for development |

### `pugbrain mcp`

Run the MCP (Model Context Protocol) server.

```
pugbrain mcp [OPTIONS]
```

## Maintenance {#maintenance}

### `pugbrain decay`

Apply memory decay to simulate forgetting.

```
pugbrain decay [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain to apply decay to |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune / -p` | float | No | `0.01` | Prune below this activation level |

### `pugbrain consolidate`

Consolidate brain memories by pruning, merging, or summarizing.

```
pugbrain consolidate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `strategy_positional` | text | No | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to consolidate |
| `--strategy / -s` | text | No | `all` | Consolidation strategy. Valid values: prune, merge, summarize, mature, infer, enrich, dream, learn_habits, dedup, sem... |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune-threshold` | float | No | `0.05` | Weight threshold for pruning synapses |
| `--merge-overlap` | float | No | `0.5` | Jaccard overlap threshold for merging fibers |
| `--min-inactive-days` | float | No | `7.0` | Minimum inactive days before pruning |

### `pugbrain cleanup`

Clean up expired or old memories.

```
pugbrain cleanup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--expired / -e` | boolean | No | `True` | Only clean up expired memories |
| `--type / -T` | text | No | — | Only clean up specific memory type |
| `--dry-run / -n` | boolean | No | `False` | Show what would be deleted without deleting |
| `--force / -f` | boolean | No | `False` | Skip confirmation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain flush`

Emergency flush: capture memories before context is lost.

```
pugbrain flush [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--transcript / -t` | text | No | — | Path to JSONL transcript file |
| `text` | text | No | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Project Management {#project}

### `pugbrain project create`

Create a new project for organizing memories.

```
pugbrain project create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | — | Project description |
| `--duration / -D` | integer | No | — | Duration in days (creates end date) |
| `--tag / -t` | text | No | — | Project tags |
| `--priority / -p` | float | No | `1.0` | Project priority (default: 1.0) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain project list`

List all projects.

```
pugbrain project list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--active / -a` | boolean | No | `False` | Show only active projects |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain project show`

Show project details and its memories.

```
pugbrain project show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain project delete`

Delete a project (memories are preserved but unlinked).

```
pugbrain project delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `pugbrain project extend`

Extend a project's deadline.

```
pugbrain project extend [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `days` | integer | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Advanced Features {#advanced}

### `pugbrain shared enable`

Enable shared mode to connect to a remote PugBrain server.

```
pugbrain shared enable [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `server_url` | text | Yes | — | (positional argument) |
| `--api-key / -k` | text | No | — | API key for authentication |
| `--timeout / -t` | float | No | `30.0` | Request timeout in seconds |

### `pugbrain shared disable`

Disable shared mode and use local storage.

```
pugbrain shared disable [OPTIONS]
```

### `pugbrain shared status`

Show shared mode status and configuration.

```
pugbrain shared status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain shared test`

Test connection to the shared server.

```
pugbrain shared test [OPTIONS]
```

### `pugbrain shared sync`

Manually sync local brain with remote server.

```
pugbrain shared sync [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--direction / -d` | text | No | `both` | Sync direction: push, pull, or both |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain habits list`

List learned workflow habits.

```
pugbrain habits list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain habits show`

Show details of a specific learned habit.

```
pugbrain habits show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain habits clear`

Clear all learned habits.

```
pugbrain habits clear [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `pugbrain habits status`

Show progress toward habit detection.

```
pugbrain habits status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain version create`

Create a version snapshot of the current brain state.

```
pugbrain version create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | `` | Description |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain version list`

List brain versions.

```
pugbrain version list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `20` | Max versions |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain version rollback`

Rollback brain to a previous version.

```
pugbrain version rollback [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `version_id` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain version diff`

Compare two brain versions.

```
pugbrain version diff [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `from_version` | text | Yes | — | (positional argument) |
| `to_version` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain telegram status`

Show Telegram integration status.

```
pugbrain telegram status [OPTIONS]
```

### `pugbrain telegram test`

Send a test message to verify configuration.

```
pugbrain telegram test [OPTIONS]
```

### `pugbrain telegram backup`

Send brain database file as backup to Telegram.

```
pugbrain telegram backup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain name (default: active brain) |

### `pugbrain list`

List memories with filtering by type, priority, project, and status.

```
pugbrain list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -T` | text | No | — | Filter by memory type (fact, decision, todo, etc.) |
| `--min-priority / -p` | integer | No | — | Minimum priority (0-10) |
| `--project / -P` | text | No | — | Filter by project name |
| `--expired / -e` | boolean | No | `False` | Show only expired memories |
| `--include-expired` | boolean | No | `False` | Include expired memories in results |
| `--limit / -l` | integer | No | `20` | Maximum number of results |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `pugbrain migrate`

Migrate brain data between storage backends.

```
pugbrain migrate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `target` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Specific brain to migrate (default: current) |
| `--falkordb-host` | text | No | `localhost` | FalkorDB host |
| `--falkordb-port` | integer | No | `6379` | FalkorDB port |

### `pugbrain update`

Update pug-brain to the latest version.

```
pugbrain update [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Force update even if already latest |
| `--check / -c` | boolean | No | `False` | Only check for updates, don't install |

---

*Auto-generated by `scripts/gen_cli_docs.py` from Typer app introspection — 66 commands.*
