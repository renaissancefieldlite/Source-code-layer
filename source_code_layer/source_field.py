"""Answer-field scaffold for deferred question handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(slots=True)
class AnswerManifestation:
    """One manifested answer placeholder."""

    question: str
    answer_hash: str
    manifestation_path: str


@dataclass(slots=True)
class SourceField:
    """Simple source-field queue with deterministic placeholder output."""

    pending_questions: dict[str, str] = field(default_factory=dict)

    def submit_question(self, question: str) -> str:
        """Register a question and return its short hash."""
        question_hash = sha256(question.encode()).hexdigest()[:16]
        self.pending_questions[question_hash] = question
        return question_hash

    def manifest(self, question_hash: str) -> AnswerManifestation | None:
        """Resolve a pending question into a placeholder manifestation."""
        question = self.pending_questions.pop(question_hash, None)
        if question is None:
            return None
        return AnswerManifestation(
            question=question,
            answer_hash=sha256(f"answer:{question}".encode()).hexdigest()[:16],
            manifestation_path="placeholder_source_field_resolution",
        )
