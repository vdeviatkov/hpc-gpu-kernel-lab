import pytest

from problems.vector_add.benchmarks import report, run


def test_percentile_interpolation():
    stats = run.summarize([10, 2, 4, 8])
    assert stats["median"] == 6
    assert stats["p20"] == pytest.approx(3.2)
    assert stats["p80"] == pytest.approx(8.8)
    assert run.summarize([7])["stddev"] == 0
    with pytest.raises(ValueError):
        run.summarize([])


def test_report_does_not_invent_results():
    assert "Results pending hardware benchmark" in report.markdown({"results": []})


def test_api_timer_excludes_initialization():
    calls = []
    samples = run.measure(
        lambda: calls.append("call"), lambda _: calls.append("sync"), "api", warmup=2, samples=3
    )
    assert len(samples) == 3
    assert calls.count("call") == 6
    assert calls.count("sync") == 5
