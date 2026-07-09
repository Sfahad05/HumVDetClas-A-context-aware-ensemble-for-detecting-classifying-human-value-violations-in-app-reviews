from humvdetclas.model import classify_review


def test_detects_safety_violation_for_harmful_content():
    prediction = classify_review("This app has harmful and dangerous challenges.")
    assert prediction.violated is True
    assert prediction.label == "safety"


def test_returns_no_violation_for_neutral_review():
    prediction = classify_review("Useful app with a clean interface.")
    assert prediction.violated is False
    assert prediction.label == "no_violation"


def test_context_boosts_relevant_label():
    prediction = classify_review(
        "The app is offensive.", context="for kids", keyword_weight=0.6, context_weight=0.4
    )
    assert prediction.violated is True
    assert prediction.label in {"respect", "safety"}
