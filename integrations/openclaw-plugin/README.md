# PugBrain OpenClaw Plugin

Brain-inspired persistent memory for AI agents in OpenClaw.

## Overview

This plugin integrates [PugBrain](https://github.com/tannht/pug-brain) with OpenClaw, providing persistent memory capabilities for AI agents. PugBrain uses a neural network-inspired architecture with neurons, synapses, and fibers to store and retrieve memories contextually.

## Prerequisites

- **Python 3.10+** with `pug-brain` installed:
  ```bash
  pip install pug-brain
  ```

- **OpenClaw** installed and configured

## Installation

### Option 1: Install from npm

```bash
npm install pug-brain-openclaw
```

### Option 2: Install from source

```bash
cd integrations/openclaw-plugin
npm install
npm run build
```

## Configuration

### 1. Enable the Plugin

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "pug-brain"
    }
  }
}
```

This replaces the default memory provider with PugBrain.

### 2. Configure Plugin Settings

```json
{
  "plugins": {
    "slots": {
      "memory": "pug-brain"
    },
    "pug-brain": {
      "pythonPath": "python",
      "brain": "default",
      "autoContext": true,
      "autoCapture": true,
      "contextDepth": 1,
      "maxContextTokens": 500,
      "timeout": 30000
    }
  }
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pythonPath` | string | `"python"` | Path to Python executable with pug-brain installed |
| `brain` | string | `"default"` | Brain name to use for this workspace |
| `autoContext` | boolean | `true` | Auto-inject relevant memories before each agent run |
| `autoCapture` | boolean | `true` | Auto-extract and store memories after each agent run |
| `contextDepth` | integer | `1` | Recall depth: 0=instant, 1=context, 2=habit, 3=deep |
| `maxContextTokens` | integer | `500` | Maximum tokens for auto-context injection |
| `timeout` | integer | `30000` | MCP request timeout in milliseconds |

### Context Depth Levels

| Level | Mode | Description |
|-------|------|-------------|
| 0 | Instant | Quick lookup of recent memories |
| 1 | Context | Contextual recall based on current task |
| 2 | Habit | Include habit patterns and workflows |
| 3 | Deep | Full graph traversal for comprehensive recall |

## Available Tools

The plugin exposes these memory tools to OpenClaw agents:

| Tool | Description |
|------|-------------|
| `pugbrain_remember` | Store a new memory with type, priority, and tags |
| `pugbrain_recall` | Search and retrieve relevant memories |
| `pugbrain_context` | Get context-aware memory suggestions |
| `pugbrain_todo` | Manage task-related memories |
| `pugbrain_auto` | Automatic memory processing |
| `pugbrain_suggest` | Get AI-powered memory suggestions |
| `pugbrain_session` | Session-based memory management |
| `pugbrain_stats` | View brain statistics |
| `pugbrain_health` | Check brain health and connectivity |
| `pugbrain_forget` | Remove outdated or incorrect memories |
| `pugbrain_edit` | Modify existing memories |

## Memory Types

PugBrain supports various memory types for different purposes:

- **fact** - Factual information
- **decision** - Choices made and rationale
- **error** - Bugs, issues, and fixes
- **insight** - Patterns and discoveries
- **preference** - User preferences
- **workflow** - Process documentation
- **instruction** - Standing instructions

## Usage Examples

### Store a Decision

```typescript
// Agent automatically captures this after task completion
{
  type: "decision",
  content: "Chose PostgreSQL over MongoDB because ACID needed for payments",
  priority: 7,
  tags: ["database", "payments", "architecture"]
}
```

### Recall Context

```typescript
// Before starting a task, agent retrieves relevant context
pugbrain_recall({
  query: "authentication implementation",
  limit: 5
})
```

### Track Workflows

```typescript
pugbrain_remember({
  type: "workflow",
  content: "Deploy process: build → test → push → verify",
  priority: 6,
  tags: ["deploy", "ci-cd"]
})
```

## Multiple Brains

Each workspace can use a different brain for isolated memory:

```json
{
  "plugins": {
    "pug-brain": {
      "brain": "project-alpha"
    }
  }
}
```

Brains are automatically created when first accessed.

## Development

### Build

```bash
npm run build
```

### Test

```bash
npm test
```

### Type Check

```bash
npm run typecheck
```

## Architecture

```
src/
  index.ts       - Plugin entry point and OpenClaw integration
  mcp-client.ts  - MCP protocol client for communicating with pug-brain
  tools.ts       - Tool definitions and implementations
  types.ts       - TypeScript type definitions
```

## Troubleshooting

### "Python not found"

Ensure `pythonPath` points to a Python installation with `pug-brain`:

```bash
python -c "import neural_memory; print(neural_memory.__version__)"
```

### "MCP connection timeout"

Increase the timeout setting:

```json
{
  "plugins": {
    "pug-brain": {
      "timeout": 60000
    }
  }
}
```

### "Brain not found"

Brains are created automatically. Check the brain name matches:

```bash
pug list  # List all available brains
```

## License

MIT

## Links

- [PugBrain Repository](https://github.com/tannht/pug-brain)
- [OpenClaw Documentation](https://github.com/openclaw/openclaw)
- [Report Issues](https://github.com/tannht/pug-brain/issues)
