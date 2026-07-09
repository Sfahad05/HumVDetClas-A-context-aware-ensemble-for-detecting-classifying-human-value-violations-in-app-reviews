from __future__ import annotations

import argparse
import json

from .model import classify_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify app reviews for human value violations."
    )
    parser.add_argument("--review", required=True, help="Review text.")
    parser.add_argument("--context", default="", help="Optional review context.")
    args = parser.parse_args()

    prediction = classify_review(args.review, args.context)
    print(
        json.dumps(
            {
                "violated": prediction.violated,
                "label": prediction.label,
                "confidence": prediction.confidence,
                "scores": prediction.scores,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
