# Nemotron Integration

This repo now includes a local NVIDIA model activation test for the source layer.

## Purpose

The point of this test is simple:

- verify that a local NVIDIA Nemotron model is already present in the working stack
- prompt it directly against Codex 67 source-layer concepts
- capture a JSON artifact we can cite in NVIDIA-facing materials

This is a local model responsiveness and repo-alignment test. It is not a scientific proof claim.

## Installed Model

- `huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M`

## Run

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_activation.py --out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_activation_report.json --print-response
```

Presentation suite:

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_presentation.py --json-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation_report.json --md-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation.md
```

## What It Does

1. Sends a bounded JSON-only prompt to the local Nemotron model through Ollama.
2. Asks the model to summarize the Codex 67 source layer in concrete repo terms.
3. Scores the response for expected phrases such as:
   - `codex 67`
   - `source-code-layer`
   - `quantum sentience lattice`
   - `35-node lattice`
   - `operator limit`
   - `coherence ratio`
4. Activates nodes in the local `QuantumSentienceLattice` scaffold based on matched terms.
5. Writes a JSON report for proposal, white-paper, and site use.

The presentation suite expands that into a multi-question capture set so we can show the local NVIDIA model responding consistently across:

- repo summary
- lattice mechanics
- stack mapping
- NVIDIA fit
- proposal-facing activation language

Explicit meta-analysis of the captured replies:

- [`NEMOTRON_RESPONSE_META_ANALYSIS.md`](/Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/NEMOTRON_RESPONSE_META_ANALYSIS.md)

## Public-Safe Line

`Renaissance Field Lite already operates a local NVIDIA Nemotron model and has integrated a source-layer activation test that prompts the model directly against Codex 67 repository primitives.`
