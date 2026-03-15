# Installation

## Requirements

- Python 3.11 or higher
- pip or pipx

## Basic Installation

Install PugBrain from PyPI:

```bash
pip install pug-brain
```

This installs the core library with CLI support.

## Optional Dependencies

PugBrain has optional features you can install as needed:

### Server (FastAPI + Web UI)

```bash
pip install pug-brain[server]
```

Includes:

- FastAPI REST API
- Interactive Web UI for brain visualization
- WebSocket sync support

### Vietnamese NLP

```bash
pip install pug-brain[nlp-vi]
```

Includes:

- underthesea - Vietnamese NLP toolkit
- pyvi - Vietnamese word segmentation

### English NLP

```bash
pip install pug-brain[nlp-en]
```

Includes:

- spaCy - Industrial NLP
- en_core_web_sm model

### Neo4j Storage

```bash
pip install pug-brain[neo4j]
```

For production deployments with Neo4j graph database.

### All Features

```bash
pip install pug-brain[all]
```

Installs everything above.

## Development Installation

For contributing or local development:

```bash
git clone https://github.com/nhadaututtheky/pug-brain
cd pug-brain
pip install -e ".[dev]"
pre-commit install
```

## Verify Installation

```bash
# Check CLI
pugbrain --version

# Check help
pugbrain --help

# Check brain status
pugbrain brain list
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_MEMORY_DIR` | Data directory | `~/.pug-brain` |
| `NEURAL_MEMORY_BRAIN` | Default brain name | `default` |
| `NEURAL_MEMORY_JSON` | Always output JSON | `false` |

## Troubleshooting

### Command not found

If `pugbrain` is not found after installation:

```bash
# Check if scripts are in PATH
python -m neural_memory.cli --help

# Or use pipx for isolated install
pipx install pug-brain
```

### Permission errors on Windows

Run terminal as Administrator or install with `--user`:

```bash
pip install --user pug-brain
```

### Missing dependencies

If optional features fail:

```bash
# Reinstall with all dependencies
pip install --force-reinstall pug-brain[all]
```
