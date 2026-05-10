# kestrel-feature-reflection

Agent self-reflection and self-improvement for Kestrel Sovereign.

## Installation

```bash
uv pip install kestrel-feature-reflection
```

The package registers `ReflectionFeature` through the `kestrel_sovereign.features`
entry point group.

## Development

```bash
uv sync --extra test
uv run --extra test pytest
```
