from src.parse import parse_verdict


def test_basic_verdicts():
    assert parse_verdict("... my verdict is [[A]]") == "model_a"
    assert parse_verdict("[[B]]") == "model_b"
    assert parse_verdict("both equal [[C]]") == "tie"


def test_last_match_wins():
    assert parse_verdict('respond "[[A]]" or "[[B]]" ... verdict: [[B]]') == "model_b"


def test_no_verdict_is_none():
    assert parse_verdict("Assistant A is better") is None
    assert parse_verdict("[A]") is None
