---
name: recap
description: Load session context and recent memories at the start of a session
arguments:
  - name: topic
    description: Optional topic to focus recall on
    required: false
allowed-tools:
  - mcp__pugbrain_recap
  - mcp__pugbrain_recall
  - mcp__pugbrain_session
---

# /recap — Session Context Loader

## Instruction

Load session context to resume where the user left off: $ARGUMENTS

## Method

### Step 1: Load Session State

```
pugbrain_session(action="get")
```

Check if there's an active session with task, feature, and progress info.

### Step 2: Load Recent Context

```
pugbrain_recap(level=2)
```

Get project state, recent decisions, progress, and active TODOs.

### Step 3: Topic-Specific Recall (if topic provided)

If the user specified a topic:

```
pugbrain_recall(query="{topic}", depth=1)
```

Retrieve memories specifically about that topic.

### Step 4: Present Summary

Format a concise session briefing:

```
Session Recap
─────────────
Project: {project_name}
Last session: {last_activity}
Progress: {current_task} ({progress}%)

Recent decisions:
  - {decision_1}
  - {decision_2}

Active TODOs:
  - {todo_1}
  - {todo_2}

{topic_section if topic provided}
```

## Rules

- Keep the summary concise — max 15 lines for the default view
- If topic is provided, dedicate a section to topic-specific memories
- If no session state exists, say so and suggest `/intake` to start capturing
- If no memories exist at all, welcome the user and explain NeuralMemory basics
- Present information in order of relevance, not chronologically
