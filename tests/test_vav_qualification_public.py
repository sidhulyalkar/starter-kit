"""Public-matrix and unexercised-actuator controls for G4 research."""
from copy import deepcopy

import pytest

from scripts.assess_vav_qualification import assess_reports
from scripts.run_vav_interventions import paired_intervention_report


def row(name, total_reward=-2.0, steps=288, modified=0):
    return {
        "arm": name, "modified_action_steps": modified, "total_steps": steps,
        "total_reward": total_reward, "loss_per_step": -total_reward / steps,
        "target_zone_shocks_ge_2_kelvin": 0,
        "all_zone_shocks_ge_2_kelvin": 0,
        "all_zone_shocks_ge_5_kelvin": 0,
        "all_zone_comfort_excursion_rate": 0.01,
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


def report():
    return {
        "scenario": "building_train_1/singapore_8",
        "sham_reproduces_reference": True,
        "arms": [row("pid_water"), row("vav1_sham"), row("vav1_reheat50")],
    }


def test_no_effect_warm_scenario_can_be_valid_evidence():
    d = assess_reports([report()], expected_scenarios=1, steps=288,
                       allow_inactive=True)
    assert d["scenario_check_results_all_pass"] is True
    assert d["per_scenario"][0]["treatment_exercised"] is False
    assert d["mean_loss_reduction_pct"] == pytest.approx(0.0)
    assert "Not qualified" in d["qualification"]


def test_unexercised_treatment_rejected_by_default():
    with pytest.raises(ValueError, match="inactive"):
        assess_reports([report()], expected_scenarios=1, steps=288)


def test_inactive_cap_cannot_claim_benefit():
    r = report()
    r["arms"][-1]["total_reward"] += 0.01
    r["arms"][-1]["loss_per_step"] = -r["arms"][-1]["total_reward"] / 288
    with pytest.raises(ValueError, match="Unexercised"):
        assess_reports([r], expected_scenarios=1, steps=288,
                       allow_inactive=True)


def test_exact_published_inventory_checked():
    with pytest.raises(ValueError, match="Published scenario coverage mismatch"):
        assess_reports([report()], expected_scenarios=1, steps=288,
                       allow_inactive=True,
                       expected_inventory={"building_train_2/denver_8"})
    data = assess_reports([report()], expected_scenarios=1, steps=288,
                          allow_inactive=True,
                          expected_inventory={"building_train_1/singapore_8"})
    assert data["scenario_count"] == 1


def test_inactive_arm_in_paired_evidence_requires_identical_outcomes():
    def arm(name, reward=-2.0):
        return {
            "scenario": "building_train_1/singapore_8",
            "config_sha256": "same-hash", "zone": "VAV_1",
            "trace": [{"step": 0, "temp_t_kelvin": 293.0,
                       "temp_tp1_kelvin": 293.1}],
            "summary": {
                "steps": 1, "total_reward": reward,
                "loss_per_step": abs(reward),
                "intervention_modified_steps": 0,
                "shock_count": 0,
                "all_zone_transitions_ge_2_kelvin": 0,
                "all_zone_transitions_ge_5_kelvin": 0,
                "all_zone_comfort_excursions_ge_2_kelvin": 0,
                "all_zone_comfort_excursion_rate": 0.0,
                "max_absolute_delta_kelvin": 0.1,
                "reward_component_sums": {
                    "normalized_productivity_regret": -1.0,
                    "normalized_energy_cost": 2.0,
                    "normalized_carbon_emission": 3.0,
                },
            },
        }
    data = {
        "pid_water": arm("pid_water"),
        "vav1_sham": arm("vav1_sham"),
        "vav1_reheat50": arm("vav1_reheat50"),
    }
    with pytest.raises(ValueError, match="No actuator"):
        paired_intervention_report(data)
    out = paired_intervention_report(data, allow_inactive_treatment=True)
    assert next(r for r in out["arms"] if r["arm"] == "vav1_reheat50")["treatment_exercised"] is False
    data["vav1_reheat50"]["summary"]["total_reward"] = -1.8
    with pytest.raises(ValueError, match="Inactive treatment"):
        paired_intervention_report(data, allow_inactive_treatment=True)
