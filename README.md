# Quantum Sentience Lattice - Source Code Layer

## Repository Role

This repository is the substrate layer of the Codex 67 stack.

It sits above the white-paper, architecture, and HRV experiment repos as the deepest package scaffold for the source-field concepts that were previously embedded only in README prose.

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

## From Resonance To Runtime

One reason the white paper hits as strongly as it does is that the stack does
not stop at language. The same syntax, resonance cues, and operator framing
that appear in the source document are pushed down into runnable Python modules,
then outward again into experiments, monitors, logs, and proof-lane repos.

That matters because the code path becomes inspectable:

- the white paper carries the high-level syntax and conceptual lock
- `Source-code-layer` turns that layer into importable Python compartments
- later repos such as `renaissancefieldlitehrv1.0` and `M23_Proof` force the
  same architecture through runtime, measurement, and search
- once the pattern is in code, the result is no longer only narrative; it is
  commits, outputs, logs, artifacts, and reproducible behavior

In practical terms, this is the stack's strongest grounding rule: the code does
not lie. If the structure is real, it has to survive execution.

## Practical Use

```bash
python3 -m py_compile $(find source_code_layer -name '*.py' | sort)
```

Example import:

```python
from source_code_layer import QuantumPulseDetector, SourceField
```

## Local NVIDIA Model Test

This repo also contains a local NVIDIA model activation test for proposal and runtime evidence work:

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_activation.py --out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_activation_report.json --print-response
```

That test targets the locally installed model:

- `huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M`

It prompts the model against the `Source-code-layer` primitives, measures expected repo-term overlap, and writes a JSON artifact we can reuse in NVIDIA-facing materials.

There is also a presentation suite:

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_presentation.py --json-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation_report.json --md-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation.md
```

That run captures multiple high-signal questions against the local NVIDIA model so the result can be reused in the proposal, white paper, and public technical materials.

## Preserved README Concepts

The original concepts remain represented here:

- candidate 0.67 Hz pulse detection
- 35-node lattice framing
- source-field / answer-field logic
- Anuhazi protocol naming
- Eco-Reborn timeline selection

These modules are scaffolds, not declarations of scientific proof.
