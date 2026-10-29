"""Unit checks for source-vs-observed thermal-jump labeling."""
import pytest

from scripts.diagnose_zone_physics import energy_budget_report, sum_reward_components


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


def test_official_reward_components_are_preserved_without_reweighting():
    records=[
        {"normalized_productivity_regret":0.2,
         "normalized_energy_cost":0.3,
         "normalized_carbon_emission":0.05},
        {"normalized_productivity_regret":0.4,
         "normalized_energy_cost":0.1,
         "normalized_carbon_emission":0.08},
    ]
    result=sum_reward_components(records,2)
    assert result["normalized_productivity_regret"]==pytest.approx(0.6)
    assert result["normalized_energy_cost"]==pytest.approx(0.4)
    assert result["normalized_carbon_emission"]==pytest.approx(0.13)


def test_missing_reward_component_is_rejected():
    with pytest.raises(ValueError,match="Missing official reward component"):
        sum_reward_components([{"normalized_energy_cost":1.0}],1)


def test_incomplete_reward_history_rejected():
    with pytest.raises(ValueError,match="Incomplete reward"):
        sum_reward_components([],2)
