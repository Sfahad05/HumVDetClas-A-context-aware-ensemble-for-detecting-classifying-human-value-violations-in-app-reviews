from __future__ import annotations

from dataclasses import dataclass

LABELS = ("no_violation", "privacy", "safety", "fairness", "respect")

KEYWORDS = {
    "privacy": {"steal", "tracking", "spy", "privacy", "data leak"},
    "safety": {"harmful", "unsafe", "violent", "dangerous"},
    "fairness": {"bias", "unfair", "discrimination", "racist", "sexist"},
    "respect": {"hate", "abuse", "insult", "harass", "offensive"},
}

CONTEXT_BOOSTS = {
    "for kids": {"safety": 0.35, "respect": 0.1},
    "education": {"fairness": 0.1, "respect": 0.1},
    "healthcare": {"privacy": 0.2, "safety": 0.2},
}


@dataclass(frozen=True)
class Prediction:
    violated: bool
    label: str
    confidence: float
    scores: dict[str, float]


def _keyword_model(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores = {label: 0.0 for label in LABELS}
    for label, words in KEYWORDS.items():
        for word in words:
            if word in lowered:
                scores[label] += 1.0
    if all(scores[label] == 0.0 for label in LABELS if label != "no_violation"):
        scores["no_violation"] = 1.0
    return scores


def _context_model(context: str) -> dict[str, float]:
    lowered = context.lower().strip()
    scores = {label: 0.0 for label in LABELS}
    boosts = CONTEXT_BOOSTS.get(lowered, {})
    for label, boost in boosts.items():
        scores[label] = boost
    return scores


def classify_review(
    review: str,
    context: str = "",
    *,
    keyword_weight: float = 0.8,
    context_weight: float = 0.2,
    threshold: float = 0.45,
) -> Prediction:
    keyword_scores = _keyword_model(review)
    context_scores = _context_model(context)

    final_scores = {}
    for label in LABELS:
        final_scores[label] = (
            keyword_weight * keyword_scores[label] + context_weight * context_scores[label]
        )

    total = sum(final_scores.values()) or 1.0
    normalized = {label: score / total for label, score in final_scores.items()}
    label = max(normalized, key=normalized.get)
    confidence = normalized[label]
    violated = label != "no_violation" and confidence >= threshold

    if not violated:
        label = "no_violation"
        confidence = normalized["no_violation"]

    return Prediction(
        violated=violated,
        label=label,
        confidence=round(confidence, 4),
        scores={k: round(v, 4) for k, v in normalized.items()},
    )
