import pytest

from lab import profile


def test_parse_value_types():
    assert profile.parse_value("100000001") == 100000001
    assert profile.parse_value("0.95") == 0.95
    assert profile.parse_value("true") is True
    assert profile.parse_value("False") is False
    assert profile.parse_value("uniform") == "uniform"


def test_build_case_overrides_and_adds_keys():
    bench = [{"n": 1, "dist": "mixed"}, {"n": 25, "dist": "mixed"}]
    assert profile.build_case(bench, []) == {"n": 25, "dist": "mixed"}
    assert profile.build_case(bench, ["n=7"], index=0) == {"n": 7, "dist": "mixed"}
    assert profile.build_case(bench, ["causal=true"]) == {"n": 25, "dist": "mixed", "causal": True}
    assert bench[-1] == {"n": 25, "dist": "mixed"}  # the problem's cases are not modified


def test_build_case_rejects_malformed_overrides():
    with pytest.raises(ValueError):
        profile.build_case([{"n": 1}], ["n"])
    with pytest.raises(ValueError):
        profile.build_case([{"n": 1}], ["=3"])
