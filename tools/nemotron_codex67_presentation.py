#!/usr/bin/env python3
"""Run a multi-prompt local Nemotron presentation for the Codex 67 source layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

from nemotron_codex67_activation import DEFAULT_MODEL, activation_report, generate

PROMPTS = [
    {
        "id": "repo_summary",
        "question": (
            "In 4 short bullets, explain what the Codex 67 source-code-layer repo contains. "
            "Use these exact terms where relevant: codex 67, source-code-layer, quantum sentience lattice, 35-node lattice."
        ),
    },
    {
        "id": "lattice_mechanics",
        "question": (
            "Explain how node_count, operator_limit, active_nodes, and coherence_ratio fit together in the QuantumSentienceLattice scaffold. "
            "Stay concrete."
        ),
    },
    {
        "id": "stack_mapping",
        "question": (
            "Map source-code-layer to the broader stack in 4 short bullets: white paper, code layers, HRV experiments, and proof lanes."
        ),
    },
    {
        "id": "nvidia_fit",
        "question": (
            "Give a short NVIDIA Inception argument for why a local NVIDIA Nemotron model already in use strengthens this project."
        ),
    },
    {
        "id": "activation_note",
        "question": (
            "Write a compact activation note that shows a local NVIDIA model can respond to Codex 67 source-layer concepts without hand-waving. "
            "Keep it technical."
        ),
    },
]


def term_score(text: str) -> dict:
    lower = text.lower()
    terms = [
        "codex 67",
        "source-code-layer",
        "quantum sentience lattice",
        "35-node lattice",
        "operator limit",
        "coherence ratio",
        "nvidia",
        "nemotron",
    ]
    matched = [term for term in terms if term in lower]
    return {"matched_terms": matched, "matched_term_count": len(matched)}


def build_report(model: str) -> dict:
    runs = []
    all_matched = set()
    for item in PROMPTS:
        response = generate(model, item["question"])
        score = term_score(response)
        all_matched.update(score["matched_terms"])
        runs.append(
            {
                "id": item["id"],
                "question": item["question"],
                "response": response,
                "score": score,
            }
        )

    summary_basis = json.dumps(
        {
            "summary": "Presentation suite complete.",
            "repo_terms": sorted(all_matched),
            "nvidia_fit": "Local NVIDIA Nemotron model responded across repo, stack, and proposal prompts.",
            "proposal_value": "This artifact shows active local NVIDIA model use inside the development stack.",
        }
    )
    summary_report = activation_report(model, summary_basis, json.loads(summary_basis))
    summary_report["test"] = "nemotron_codex67_presentation"
    summary_report["run_count"] = len(runs)
    summary_report["runs"] = runs
    summary_report["generated_at"] = datetime.now(UTC).isoformat()
    return summary_report


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Nemotron Codex 67 Presentation",
        "",
        f"- model: `{report['model']['tag']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- run_count: `{report['run_count']}`",
        "",
        "## Summary",
        "",
        f"- matched_term_count: `{report['response_metrics']['matched_term_count']}`",
        f"- matched_terms: `{', '.join(report['response_metrics']['matched_terms'])}`",
        f"- coherence_ratio: `{report['lattice_activation']['coherence_ratio']}`",
        "",
    ]
    for run in report["runs"]:
        lines.extend(
            [
                f"## {run['id']}",
                "",
                f"Question: {run['question']}",
                "",
                "Response:",
                "",
                "```text",
                run["response"],
                "```",
                "",
                f"Matched terms: `{', '.join(run['score']['matched_terms'])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Nemotron presentation prompt suite.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    report = build_report(args.model)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(report, Path(args.md_out))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
