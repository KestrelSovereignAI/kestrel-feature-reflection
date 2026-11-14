# kestrel-feature-reflection

Agent self-reflection and self-improvement for Kestrel Sovereign.

## Installation

```bash
uv pip install kestrel-feature-reflection
```

The package registers `ReflectionFeature` through the `kestrel_sovereign.features`
entry point group.

## Conversation and memory boundaries

Version 0.4 requires Core 0.66.0 or newer. Reflection uses Core's canonical
speaker projection and escaped input boundaries. Connected clients' external
assistant transcripts remain quoted external data, rather than Kestrel's own
responses or actions. Memory excerpts are quoted as data as well.

Conversation reads go through the live privacy-aware conversation owner.
Reflection sessions and insights use MemorySystem's existing persistence
permission and hold its privacy-transition lock through both original writes.
EPHEMERAL, ISOLATED and DEIDENTIFIED modes do not persist these derivatives.

Tests cover native SQLite, privacy transitions, genuine native MCP source rows
and prompt construction with generation disabled. These checks do not establish
LLM judgment quality, a vendor account installation or an overnight deployment.

## Development

```bash
uv sync --extra test
uv run --extra test pytest
```
