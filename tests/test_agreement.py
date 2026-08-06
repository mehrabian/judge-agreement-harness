"""Hand-computable fixtures. If you cannot compute the expected value on paper,
the fixture is too big."""
import pandas as pd
import pytest

from eval.agreement import agreement


def fixture_6_items() -> pd.DataFrame:
    """Six aligned verdict pairs.
    with ties:   agree = 4/6
    without ties (either judge said tie -> dropped): agree = 3/4
    kappa (with ties, 3-class): 0.52
    """
    return pd.DataFrame(
        {
            "question_id": [1, 2, 3, 4, 5, 6],
            "model_a": ["alpha"] * 6,
            "model_b": ["beta"] * 6,
            "verdict_1": ["model_a", "model_b", "model_a", "tie", "model_a", "model_b"],
            "verdict_2": ["model_a", "model_b", "model_b", "tie", "tie", "model_b"],
        }
    )


@pytest.mark.parametrize("drop_ties,expected", [(False, 4 / 6), (True, 3 / 4)])
def test_agreement_fixture(drop_ties, expected):
    res = agreement(fixture_6_items(), drop_ties=drop_ties)
    assert res["agree"] == pytest.approx(expected)


def test_kappa_fixture():
    # po=4/6; pe=11/36; kappa=(24/36-11/36)/(25/36)=13/25=0.52
    res = agreement(fixture_6_items(), drop_ties=False)
    assert res["kappa"] == pytest.approx(0.52, abs=1e-3)
