"""Proof that complete evidence and safety tradeoffs are distinct outcomes."""
from copy import deepcopy

import pytest

from scripts.assess_vav_qualification import assess_reports


def report(scenario="building_train_2/denver_8", steps=8640):
    def row(name, loss, comfort, shocks5, modified):
        return {
            "arm": name, "total_steps": steps,
            "total_reward": -loss * steps, "loss_per_step": loss,
            "modified_action_steps": modified,
            "target_zone_shocks_ge_2_kelvin": 10,
            "all_zone_shocks_ge_2_kelvin": 20,
            "all_zone_shocks_ge_5_kelvin": shocks5,
            "all_zone_comfort_excursion_rate": comfort,
            "all_zone_comfort_excursion_difference": 0,
            "reward_component_totals": {
                "normalized_productivity_regret": -1.0,
                "normalized_energy_cost": 2.0,
                "normalized_carbon_emission": 3.0,
            },
            "reward_component_differences": {
                "normalized_productivity_regret": 0.0,
                "normalized_energy_cost": 0.0,
                "normalized_carbon_emission": 0.0,
            },
        }
    return {
        "scenario": scenario, "sham_reproduces_reference": True,
        "arms": [
            row("pid_water", 0.03, 0.05, 20, 0),
            row("vav1_sham", 0.03, 0.05, 20, 0),
            row("vav1_reheat50", 0.02, 0.06, 15, 50),
        ],
    }


def test_comfort_regression_is_preserved_not_ignored():
    data = assess_reports([report()], expected_scenarios=1, steps=8640)
    assert data["mean_loss_reduction_pct"] == pytest.approx(100 / 3)
    assert data["scenario_check_results_all_pass"] is False
    assert data["decision"] == "experiment_concerning_regression_or_tradeoff"
    assert data["per_scenario"][0]["experimental_checks"]["comfort_proxy_nonworsening"] is False
    assert "Not qualified" in data["qualification"]


def test_one_day_is_not_one_month():
    with pytest.raises(ValueError, match="Truncated"):
        assess_reports([report(steps=288)], expected_scenarios=1, steps=8640)


def test_unpaired_scenario_count_rejected():
    with pytest.raises(ValueError, match="coverage"):
        assess_reports([report()], expected_scenarios=30, steps=8640)


def test_duplicate_scenario_and_sham_drift_fail_closed():
    r = report()
    with pytest.raises(ValueError, match="duplicate"):
        assess_reports([r, deepcopy(r)], expected_scenarios=2, steps=8640)
    r["arms"][1]["total_reward"] += 1
    with pytest.raises(ValueError, match="inconsistent|Sham"):
        assess_reports([r], expected_scenarios=1, steps=8640)


def test_inactive_cap_rejected():
    r = report()
    r["arms"][-1]["modified_action_steps"] = 0
    with pytest.raises(ValueError, match="inactive"):
        assess_reports([r], expected_scenarios=1, steps=8640)


def test_modified_reward_without_matched_loss_rejected():
    r = report()
    r["arms"][-1]["total_reward"] = -0.01
    with pytest.raises(ValueError, match="inconsistent"):
        assess_reports([r], expected_scenarios=1, steps=8640)


def test_component_delta_must_reconcile():
    r = report()
    r["arms"][-1]["reward_component_differences"]["normalized_energy_cost"] = -1
    with pytest.raises(ValueError, match="Incorrect"):
        assess_reports([r], expected_scenarios=1, steps=8640)


def test_sham_component_drift_rejected():
    r = report()
    r["arms"][1]["reward_component_totals"]["normalized_carbon_emission"] = 4
    with pytest.raises(ValueError, match="Sham changed"):
        assess_reports([r], expected_scenarios=1, steps=8640)


def test_clean_metrics_still_not_generalization():
    r = report()
    r["arms"][-1]["all_zone_comfort_excursion_rate"] = 0.049
    data = assess_reports([r], expected_scenarios=1, steps=8640)
    assert data["decision"] == "preliminary_constraints_passed_not_qualified"
    assert data["status"] == "valid_paired_evidence_not_controller_qualification"


def test_nan_component_rejected():
    r = report()
    r["arms"][-1]["reward_component_totals"]["normalized_energy_cost"] = float("nan")
    with pytest.raises(ValueError):
        assess_reports([r], expected_scenarios=1, steps=8640)
