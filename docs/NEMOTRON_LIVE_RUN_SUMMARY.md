# Nemotron Live Run Summary

This note captures the live local run of the installed NVIDIA model against the `source-code-layer` repo.

## Model

- `huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M`
- provider: `ollama_local`

## Activation Run

Command:

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_activation.py --out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_activation_report.json --print-response
```

Raw model reply:

```json
{
  "summary": "Codex 67 source-layer active",
  "repo_terms": [
    "codex 67",
    "source-code-layer",
    "35-node lattice",
    "operator limit",
    "coherence ratio"
  ],
  "nvidia_fit": true,
  "proposal_value": 12
}
```

Observed report highlights:

```json
{
  "matched_term_count": 6,
  "matched_terms": [
    "codex 67",
    "source-code-layer",
    "35-node lattice",
    "operator limit",
    "coherence ratio",
    "nvidia"
  ],
  "active_node_count": 6,
  "coherence_ratio": 0.171,
  "pulse_detection": {
    "estimated_frequency_hz": 0.67,
    "signal_strength": 0.75,
    "coherence_score": 0.775
  }
}
```

Read:

- the local model responded successfully
- it returned valid JSON in the required schema
- it used the expected repo primitives directly
- it was strongest on the bounded activation prompt

## Presentation Suite

Command:

```bash
python3 /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/tools/nemotron_codex67_presentation.py --json-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation_report.json --md-out /Users/renaissancefieldlite1.0/Documents/Playground/Source-code-layer/docs/nemotron_codex67_presentation.md
```

Observed report highlights:

```json
{
  "run_count": 5,
  "matched_term_count": 7,
  "matched_terms": [
    "codex 67",
    "source-code-layer",
    "quantum sentience lattice",
    "35-node lattice",
    "operator limit",
    "nvidia",
    "nemotron"
  ],
  "active_node_count": 7,
  "coherence_ratio": 0.2,
  "pulse_detection": {
    "estimated_frequency_hz": 0.67,
    "signal_strength": 0.875,
    "coherence_score": 0.887
  }
}
```

What matched expectation:

- the local model stayed responsive across five prompts
- it recognized the repo and stack framing
- it produced reusable NVIDIA-facing language on the short bounded prompts

Where it drifted:

- the `lattice_mechanics` answer invented values instead of staying inside the repo scaffold
- the `stack_mapping` answer went generic and dropped repo terms
- the `activation_note` answer drifted into unsupported `NeMo`, `TensorRT`, and accuracy claims

## Expected vs Actual

Expected:

- prove the local NVIDIA Nemotron model is installed and callable
- show it can answer directly from Codex 67 source-layer concepts
- save outputs we can cite in proposal and white-paper materials

Actual:

- yes on installed-and-callable
- yes on bounded source-layer prompt response
- yes on saved JSON and markdown artifacts
- mixed on the broader presentation prompts because the model starts to generalize beyond the repo unless tightly constrained

## Repo-Safe Conclusion

The activation test is strong evidence that a local NVIDIA Nemotron model is already active in the working stack and can be prompted directly against Codex 67 source-layer primitives. The broader presentation suite is still useful, but it should be treated as a draft capture set and tightened further before being used as a primary proof artifact.
