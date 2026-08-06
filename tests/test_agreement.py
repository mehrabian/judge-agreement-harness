"""Hand-computable fixtures. If you cannot compute the expected value on paper,
the fixture is too big."""
import pandas as pd
import pytest

from eval.agreement import agreement


def fixture_6_items() -> pd.DataFrame:
    """Six aligned verdict pairs.
    TODO(harness): build so that, by hand:
      - with ties:   agree = 4/6
      - without ties (either judge said tie -> dropped): agree = 3/4
      - kappa: compute on paper from the 2x2 (or 3x3) table and assert to 3 decimals.
    """
    raise NotImplementedError


@pytest.mark.parametrize("drop_ties,expected", [(False, 4 / 6), (True, 3 / 4)])
def test_agreement_fixture(drop_ties, expected):
    res = agreement(fixture_6_items(), drop_ties=drop_ties)
    assert res["agree"] == pytest.approx(expected)


def test_kappa_fixture():
    res = agreement(fixture_6_items(), drop_ties=False)
    assert res["kappa"] == pytest.approx(None, abs=1e-3)  # TODO(harness): paper value here
