# 🐶 PugBrain: Reflex-based Neural Memory

> **Reflex-based memory system for AI agents — retrieval through activation, not search.**

PugBrain is a hybrid neural-vector memory system designed for long-term agent continuity. It combines graph relationships with vector embeddings to simulate "reflexive" recall.

## 🚀 Quick Start

```bash
pip install "pug-brain[server]"
```

## 🛠 Features
- **Hybrid Retrieval:** Neural graph + Vector similarity.
- **Built-in Dashboard:** Visualize memory clusters and agent habits.
- **MCP Support:** Ready for Claude Desktop and OpenClaw.
- **Pluggable Backends:** SQLite (default), Neo4j, FalkorDB, ChromaDB.

## 🖥 Dashboard
Serve the memory system with a built-in UI:
```bash
pugbrain serve --port 18790
```
Visit: `http://localhost:18790/ui`

## 🧠 Core Philosophy
Memory shouldn't be a database query; it should be an activation signal. PugBrain treats entities as neurons that "fire" based on context.

---
*Created by PugBrain Contributors. Woof! 🐶*
