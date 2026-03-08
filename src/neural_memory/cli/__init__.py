"""PugBrain CLI.

Simple command-line interface for storing and retrieving memories.

Usage:
    pug remember "content"     Store a memory
    pug recall "query"         Query memories
    pug context                Get recent context
    pug brain list             List brains
    pug brain use <name>       Switch brain
"""

from neural_memory.cli.main import app, main

__all__ = ["app", "main"]
