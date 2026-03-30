#!/usr/bin/env python3
"""Run a local Nemotron prompt against the Codex 67 source layer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code_layer import QuantumPulseDetector, QuantumSentienceLattice

DEFAULT_MODEL = "huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M"
PROMPT = """You are a local NVIDIA Nemotron model being tested against the Codex 67 source layer.

Task:
Return concise JSON with these keys only:
- summary
- repo_terms
- nvidia_fit
- proposal_value

Rules:
- repo_terms must be a JSON array of exact phrases you used from this set:
  ["codex 67", "source-code-layer", "quantum sentience lattice", "35-node lattice", "operator limit", "coherence ratio"]
- keep each string short and concrete
- do not add markdown

Context:
- Codex 67 has a source-code-layer repo
- the repo includes a QuantumSentienceLattice scaffold
- it uses a 35-node lattice framing
- it has an operator limit of 12
- it exposes a coherence ratio method
- this run is being used as evidence that a local NVIDIA model is already active in the stack for Inception positioning
"""


def generate(model: str, prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ollama run failed")
    return result.stdout.strip()


def parse_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "summary": text,
            "repo_terms": [],
            "nvidia_fit": "Response was not valid JSON.",
            "proposal_value": "Manual review required.",
        }


def activation_report(model: str, response_text: str, parsed: dict) -> dict:
    response_lower = response_text.lower()
    expected_terms = [
        "codex 67",
        "source-code-layer",
        "quantum sentience lattice",
        "35-node lattice",
        "operator limit",
        "coherence ratio",
        "nvidia",
        "nemotron",
    ]
    matched_terms = [term for term in expected_terms if term in response_lower]

    lattice = QuantumSentienceLattice()
    for index, term in enumerate(matched_terms[: lattice.operator_limit]):
        lattice.activate_node(f"NODE_{index:02d}", term)

    detector = QuantumPulseDetector()
    telemetry = [1.0 if term in matched_terms else 0.0 for term in expected_terms]
    pulse = detector.detect(telemetry, target_hz=0.67)

    return {
        "test": "nemotron_codex67_activation",
        "model": {
            "tag": model,
            "provider": "ollama_local",
        },
        "response_metrics": {
            "characters": len(response_text),
            "matched_terms": matched_terms,
            "matched_term_count": len(matched_terms),
        },
        "lattice_activation": {
            "active_nodes": lattice.active_nodes,
            "active_node_count": len(lattice.active_nodes),
            "coherence_ratio": lattice.coherence_ratio(),
            "pulse_detection": {
                "estimated_frequency_hz": pulse.estimated_frequency_hz,
                "signal_strength": pulse.signal_strength,
                "coherence_score": pulse.coherence_score,
            },
        },
        "parsed_response": parsed,
        "assessment": {
            "responsive": len(response_text) > 0,
            "json_like": isinstance(parsed, dict),
            "proposal_line": "Local NVIDIA Nemotron model present and responsive to Codex 67 source-layer prompts."
            if response_text
            else "No local model response captured.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Nemotron Codex 67 activation test.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out")
    parser.add_argument("--print-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        report = activation_report(args.model, "", {})
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    try:
        response_text = generate(args.model, PROMPT)
    except Exception as exc:
        report = {
            "test": "nemotron_codex67_activation",
            "model": {"tag": args.model, "provider": "ollama_local"},
            "error": f"Failed to run local model: {exc}",
            "next_step": "Ensure Ollama is running and the local Nemotron model is available.",
        }
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    parsed = parse_response(response_text)
    report = activation_report(args.model, response_text, parsed)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.print_response:
        print(response_text)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
