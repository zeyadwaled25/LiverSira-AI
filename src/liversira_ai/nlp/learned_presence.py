"""Frozen learned presence model (word TF-IDF + logistic regression)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 42


def frozen_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
