from evaluation.learned_presence.compare_learned_presence import _normalise_template


def test_template_normalization_removes_numeric_variation():
    first = _normalise_template("The liver measures 12.4 cc.")
    second = _normalise_template("The liver measures 9.1 cc.")

    assert first == second
