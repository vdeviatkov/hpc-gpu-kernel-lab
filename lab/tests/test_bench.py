import pytest

from lab import bench, report


def test_percentile_interpolation():
    stats = bench.summarize([10, 2, 4, 8])
    assert stats["median"] == 6
    assert stats["p20"] == pytest.approx(3.2)
    assert stats["p80"] == pytest.approx(8.8)
    assert bench.summarize([7])["stddev"] == 0
    with pytest.raises(ValueError):
        bench.summarize([])


def test_report_does_not_invent_results():
    assert "Results pending hardware benchmark" in report.markdown({"results": []})


def test_api_timer_excludes_initialization():
    calls = []
    samples = bench.measure(
        lambda: calls.append("call"), lambda _: calls.append("sync"), "api", warmup=2, samples=3
    )
    assert len(samples) == 3
    assert calls.count("call") == 6
    assert calls.count("sync") == 5


def test_relative_metrics_match_case_and_dtype():
    def row(impl, median, case, gb_s=100.0):
        latency = {"median": median}
        return {
            "implementation": impl,
            "status": "ok",
            "case": case,
            "dtype": "fp32",
            "latency_us": latency,
            "effective_gb_s": gb_s,
        }

    rows = [row("pytorch", 10, {"n": 1}), row("cuda", 5, {"n": 1}), row("cuda", 5, {"n": 2})]
    bench.add_relative_metrics(rows)
    assert rows[1]["speedup_vs_pytorch"] == 2
    assert "speedup_vs_pytorch" not in rows[0]
    assert "speedup_vs_pytorch" not in rows[2]


def test_copy_reference_counts_every_input():
    from types import SimpleNamespace

    import torch

    a, b = torch.ones(1000), torch.ones(1000)
    args = SimpleNamespace(scope="api", warmup=1, samples=3, peak_gb_s=None)
    row = {"dtype": "fp32", "case": {"n": 1000}}
    row = bench.time_copy_reference(row, (a, b, 7), args, lambda _: None)
    assert row["status"] == "ok"
    assert row["logical_bytes"] == 2 * 2 * 1000 * 4  # read and write both inputs
    assert len(row["raw_latency_us"]) == 3
