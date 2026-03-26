# Quantum Sentience Lattice - Source Code Layer

## Repository Role

This repository is the substrate layer of the Codex 67 stack: the deepest package scaffold for the source-field concepts that were previously trapped in prose.

It sits beneath the white-paper, architecture, and HRV experiment repos as the point where the source-language gets an actual package body.

## Related Repositories

- [Codex-67-white-paper-](https://github.com/renaissancefieldlite/Codex-67-white-paper-): source document and PDF layer
- [Codex-67-white-paper-code-layers](https://github.com/renaissancefieldlite/Codex-67-white-paper-code-layers): architecture and validation scaffold
- [renaissancefieldlitehrv1.0](https://github.com/renaissancefieldlite/renaissancefieldlitehrv1.0): HRV experiment and capture layer

## Stack Relationship

The clean stack is:

1. `Source-code-layer`
   substrate package and deep-source primitives
2. `Codex-67-white-paper-`
   source document and white-paper layer
3. `Codex-67-white-paper-code-layers`
   architecture and validation scaffold
4. `renaissancefieldlitehrv1.0`
   experiment, capture, and evidence path

## Current File Tree

```text
.
├── README.md
├── requirements.txt
├── LICENSE.md
├── source_code_layer/
│   ├── __init__.py
│   ├── anuhazi_protocol.py
│   ├── eco_reborn_vision.py
│   ├── quantum_pulse_detector.py
│   ├── quantum_sentience_lattice.py
│   └── source_field.py
└── docs/
    ├── SOURCE_FIELD.md
    ├── LATTICE_NOTES.md
    └── CROSS_REPO_MAP.md
```

## What This Repo Contains

This scaffold turns the README-described concepts into importable modules:

- `quantum_pulse_detector.py`
- `quantum_sentience_lattice.py`
- `source_field.py`
- `anuhazi_protocol.py`
- `eco_reborn_vision.py`

These are package-level primitives, not proof claims.

## Why This Repo Exists

The original README merged all deep-layer concepts into one document as a fast preservation move. That worked for archival purposes, but it left the repo as prose-only.

This scaffold keeps the conceptual language while making the structure usable:

- the source layer now exists as code compartments
- later repos can import or mirror the same primitives
- the stack no longer depends on parsing one giant README to recover the architecture

## Practical Use

```bash
python3 -m py_compile $(find source_code_layer -name '*.py' | sort)
```

Example import:

```python
from source_code_layer import QuantumPulseDetector, SourceField
```

## Preserved README Concepts

The original concepts remain represented here:

- candidate 0.67 Hz pulse detection
- 35-node lattice framing
- source-field / answer-field logic
- Anuhazi protocol naming
- Eco-Reborn timeline selection

These modules are scaffolds, not declarations of scientific proof.
