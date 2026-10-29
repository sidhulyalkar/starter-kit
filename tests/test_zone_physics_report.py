"""Unit checks for source-vs-observed thermal-jump labeling."""
import pytest

from scripts.diagnose_zone_physics import energy_budget_report


def test_shock_report_preserves_large_and_small_events():
    records = [
        {"step":0, "delta_kelvin":0.1, "source_only_delta_k_approx":0.09},
        {"step":1, "delta_kelvin":7.12, "source_only_delta_k_approx":2.8},
        {"step":2, "delta_kelvin":-3.0, "source_only_delta_k_approx":-0.2},
    ]
    report=energy_budget_report(records, threshold_k=2.0)
    assert report["steps"] == 3
    assert report["shock_count"] == 2
    assert report["shock_steps"] == [1,2]
    assert report["top_12_shocks"][0]["step"] == 1
    assert report["max_absolute_delta_kelvin"] == pytest.approx(7.12)
    assert "omit" in report["interpretation"]


@pytest.mark.parametrize("records,threshold", [([],2), ([{"step":0,"delta_kelvin":1}],0)])
def test_empty_and_invalid_threshold_rejected(records,threshold):
    with pytest.raises(ValueError):
        energy_budget_report(records,threshold)
