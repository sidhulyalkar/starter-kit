"""Preserve day-level reversals, trace integrity and unknown reward provenance."""
from copy import deepcopy

import pytest

from smart_buildings.experiments.regime_diagnostics import analyze_pair


def report(policy, days=2, cap=False):
    trace = []
    previous = 293.0
    for step in range(days * 288):
        # One cooling step at the day-1 boundary, then sustained difference.
        next_temp = 293.0 - (1.5 if cap and step >= 288 else 0.0)
        trace.append({
            "step": step, "minute": step * 5,
            "temp_t_kelvin": previous,
            "temp_tp1_kelvin": next_temp,
            "delta_kelvin": next_temp - previous,
            "realized_reheat_fraction": .5 if cap else 1.0,
            "realized_damper_fraction": .5,
            "realized_zone_air_flow_m3_s": .2,
            "source_power_w_reported_by_simulator": 100.0,
        })
        previous = next_temp
    return {
        "scenario": "building_train_2/denver_8",
        "config_sha256": "abc123",
        "zone": "VAV_1",
        "room_floor_area_m2": 60.0,
        "room_heat_capacity_j_per_kelvin": 222000.,
        "policy": policy, "trace": trace,
    }


def test_first_day_vs_subsequent_cold_offset():
    baseline = report("pid_water")
    treated = report("vav1_reheat50", cap=True)
    x = analyze_pair(baseline, treated, steps=576)
    assert x["days"] == 2
    assert x["first_day"]["mean_treatment_minus_reference_temp_kelvin"] == 0
    assert x["later_days"]["mean_temperature_gap_kelvin"] == pytest.approx(-1.5)
    assert x["all_steps"]["time_steps_treatment_at_least_1k_colder"] == 288
    assert x["all_steps"]["longest_colder_run_steps"] == 288
    assert x["per_day"][1]["change_in_mean_reheat_fraction"] == pytest.approx(-.5)
    assert "cannot be inferred" in x["unavailable_from_saved_trace"]


def test_unmatched_scenario_rejected():
    baseline = report("pid_water")
    treated = report("vav1_reheat50", cap=True)
    treated["config_sha256"] = "other"
    with pytest.raises(ValueError, match="Mismatched config_sha256"):
        analyze_pair(baseline, treated, steps=576)


def test_missing_duplicate_or_invalid_steps_fail():
    baseline = report("pid_water")
    treated = report("vav1_reheat50", cap=True)
    bad = deepcopy(treated)
    bad["trace"][1]["step"] = 0
    with pytest.raises(ValueError, match="Invalid/duplicate"):
        analyze_pair(baseline, bad, steps=576)
    bad = deepcopy(treated)
    bad["trace"].pop()
    with pytest.raises(ValueError, match="complete"):
        analyze_pair(baseline, bad, steps=576)
    bad = deepcopy(treated)
    bad["trace"][300]["minute"] = 300 * 60
    with pytest.raises(ValueError, match="Misaligned"):
        analyze_pair(baseline, bad, steps=576)


def test_no_day_level_reward_invention():
    x = analyze_pair(report("pid_water"),
                     report("vav1_reheat50", cap=True), steps=576)
    assert "reward" not in x["per_day"][0]
    assert "comfort" not in x["per_day"][0]
    with pytest.raises(ValueError, match="complete 288-step days"):
        analyze_pair(report("pid_water"), report("vav1_reheat50"),
                     steps=287)


def test_bad_actuator_or_delta_rejected():
    a = report("pid_water")
    b = report("vav1_reheat50", cap=True)
    b["trace"][0]["realized_reheat_fraction"] = 1.7
    with pytest.raises(ValueError, match="Actuator"):
        analyze_pair(a,b,steps=576)
    b = report("vav1_reheat50", cap=True)
    b["trace"][0]["delta_kelvin"] = 20.0
    with pytest.raises(ValueError, match="inconsistent"):
        analyze_pair(a,b,steps=576)
