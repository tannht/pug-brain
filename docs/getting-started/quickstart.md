# Quick Start

This guide walks you through basic PugBrain usage in 5 minutes.

!!! tip "3 tools you need"
    PugBrain has 44 tools, but you only need three: **`nmem_remember`**, **`nmem_recall`**, and **`nmem_health`**. The agent handles the other 41 automatically. See [all tools](../guides/mcp-server.md#available-tools).

## 0. Setup

### Claude Code (Plugin)

```bash
/plugin marketplace add nhadaututtheky/pug-brain
/plugin install pug-brain@pug-brain-marketplace
```

### OpenClaw (Plugin)

```bash
pip install pug-brain
npm install -g pugbrain
```

Then in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "pugbrain"
    }
  }
}
```

Restart the gateway. The plugin auto-registers 6 tools (`pugbrain_remember`, `pugbrain_recall`, `pugbrain_context`, `pugbrain_todo`, `pugbrain_stats`, `pugbrain_health`) and injects memory context before each agent run. See the [full setup guide](../guides/openclaw-plugin.md).

### Cursor / Windsurf / Other MCP Clients

```bash
pip install pug-brain
```

Then add `pugbrain-mcp` to your editor's MCP config. No `pugbrain init` needed — the MCP server auto-initializes on first use.

### VS Code Extension

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=neuralmem.pugbrain) for a visual interface — sidebar memory tree, interactive graph explorer, CodeLens on functions, and keyboard shortcuts for encode/recall.

### Optional: Explicit Init

```bash
pugbrain init    # Only needed if you want to pre-create config and brain
```

## 1. Store Your First Memory

```bash
pugbrain remember "Fixed auth bug with null check in login.py:42"
```

Output:
```
Stored memory with 4 neurons and 3 synapses
```

## 2. Query Memories

```bash
pugbrain recall "auth bug"
```

Output:
```
Fixed auth bug with null check in login.py:42
(confidence: 0.85, neurons activated: 4)
```

## 3. Use Memory Types

Different types help organize and retrieve memories:

```bash
# Decisions (never expire)
pugbrain remember "We decided to use PostgreSQL" --type decision

# TODOs (expire in 30 days)
pugbrain todo "Review PR #123" --priority 7

# Facts
pugbrain remember "API endpoint is /v2/users" --type fact

# Errors with solutions
pugbrain remember "ERROR: null pointer in auth. SOLUTION: add null check" --type error
```

## 4. Get Context

Retrieve recent memories for AI context injection:

```bash
pugbrain context --limit 5
```

With JSON output for programmatic use:

```bash
pugbrain context --limit 5 --json
```

## 5. View Statistics

```bash
pugbrain stats
```

Output:
```
Brain: default
Neurons: 12
Synapses: 18
Fibers: 4

Memory Types:
  fact: 2
  decision: 1
  todo: 1
```

## 6. Manage Brains

Create separate brains for different projects:

```bash
# List brains
pugbrain brain list

# Create new brain
pugbrain brain create work

# Switch to brain
pugbrain brain use work

# Export brain
pugbrain brain export -o backup.json
```

## 7. Web Visualization

Start the server to visualize your brain:

```bash
pip install pug-brain[server]
pugbrain serve
```

Open http://localhost:8000/ui to see:

- Interactive neural graph
- Color-coded neuron types
- Click nodes for details

## Example Workflow

Here's a typical workflow during a coding session:

```bash
# Start of session - get context
pugbrain context --limit 10

# During work - remember important things
pugbrain remember "UserService now uses async/await"
pugbrain remember "DECISION: Use JWT for auth. REASON: Stateless" --type decision
pugbrain todo "Add rate limiting to API" --priority 8

# When you need to recall
pugbrain recall "auth decision"
pugbrain recall "UserService changes"

# End of session - check what's pending
pugbrain list --type todo
```

## Next Steps

- [CLI Reference](cli.md) — All commands and options
- [Memory Types](../concepts/memory-types.md) — Understanding different memory types
- [Integration Guide](../guides/integration.md) — Integrate with Claude Code, Cursor, and other editors
- [OpenClaw Plugin Guide](../guides/openclaw-plugin.md) — Full setup for OpenClaw agents
