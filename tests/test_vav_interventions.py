"""No-aliasing VAV cap tests plus sham outcome invariants."""
import numpy as np
import pytest

from smart_buildings.experiments.vav_interventions import (
    VAVCaps, cap_vav_command, intervention_policy_class,
)
from scripts.run_vav_interventions import paired_intervention_report


def metadata():
    return {
        "zone_action_indices": {
            "VAV_1": {
                "supply_air_damper_percentage_command": 2,
                "reheat_valve_percentage_command": 3
            },
            "VAV_2": {
                "supply_air_damper_percentage_command": 4,
                "reheat_valve_percentage_command": 5
            }
        }
    }


def action():
    return {
        "continuous_action": np.asarray([-0.1, 0.8, 1.0, 1.0,
                                         0.7, -0.2, -0.9], dtype=np.float32),
        "discrete_action": np.asarray([1, 0, 1], dtype=np.int32),
    }


def test_sham_leaves_every_action_identical():
    a=action()
    out=cap_vav_command(a,metadata(),VAVCaps())
    np.testing.assert_array_equal(out["continuous_action"],a["continuous_action"])
    np.testing.assert_array_equal(out["discrete_action"],a["discrete_action"])
    assert out["continuous_action"] is not a["continuous_action"]


def test_native_fraction_maps_to_normalized_commands():
    a=action()
    out=cap_vav_command(a,metadata(),VAVCaps(damper_max_fraction=0.75,
                                              reheat_max_fraction=0.5))
    assert out["continuous_action"][2] == pytest.approx(0.5)
    assert out["continuous_action"][3] == pytest.approx(0.0)
    for i in [0,1,4,5,6]:
        assert out["continuous_action"][i] == a["continuous_action"][i]
    np.testing.assert_array_equal(out["discrete_action"],a["discrete_action"])
    assert a["continuous_action"][2] == 1.0
    assert a["continuous_action"][3] == 1.0


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_invalid_caps_rejected(bad):
    with pytest.raises(ValueError):
        VAVCaps(reheat_max_fraction=bad)


def test_bad_index_mapping_rejected():
    meta=metadata()
    meta["zone_action_indices"]["VAV_1"]["reheat_valve_percentage_command"]=2
    with pytest.raises(ValueError,match="alias"):
        cap_vav_command(action(),meta,VAVCaps(reheat_max_fraction=0.5))


def test_missing_target_rejected():
    with pytest.raises(ValueError,match="absent"):
        cap_vav_command(action(),metadata(),VAVCaps(vav_id="VAV_999"))


def test_invalid_arm_name_rejected():
    with pytest.raises(ValueError):
        intervention_policy_class("pid_water",VAVCaps())


def _report(name,loss=0.1,modified=0,shock=10,comfort=3):
    return {
        "scenario":"building_train_2/denver_8",
        "config_sha256":"same-config",
        "zone":"VAV_1", "policy":name,
        "trace":[
            {"step":0,"temp_t_kelvin":292.0,"temp_tp1_kelvin":294.0},
            {"step":1,"temp_t_kelvin":294.0,"temp_tp1_kelvin":293.0}
        ],
        "summary":{
            "steps":2,"intervention_modified_steps":modified,
            "total_reward":-2*loss,"loss_per_step":loss,
            "shock_count":shock,
            "all_zone_transitions_ge_2_kelvin":12,
            "all_zone_transitions_ge_5_kelvin":2,
            "all_zone_comfort_excursions_ge_2_kelvin":comfort,
            "all_zone_comfort_excursion_rate":0.15,
            "max_absolute_delta_kelvin":2.0,
            "reward_component_sums":{
                "normalized_productivity_regret":0.2,
                "normalized_energy_cost":0.3,
                "normalized_carbon_emission":0.4,
            },
        }
    }


def test_sham_matches_and_treatment_evaluated():
    reports={
        "pid_water":_report("pid_water"),
        "vav1_sham":_report("vav1_sham"),
        "vav1_reheat50":_report("vav1_reheat50",loss=0.09,modified=2,shock=4)
    }
    result=paired_intervention_report(reports)
    assert result["sham_reproduces_reference"] is True
    treatment=next(x for x in result["arms"] if x["arm"]=="vav1_reheat50")
    assert treatment["relative_loss_change_pct"]==pytest.approx(-10)
    assert treatment["target_zone_shock_difference"] == -6
    assert treatment["reward_component_totals"]["normalized_energy_cost"] == pytest.approx(0.3)
    assert treatment["reward_component_differences"]["normalized_energy_cost"] == pytest.approx(0.0)


def test_sham_divergence_fails_closed():
    reports={"pid_water":_report("pid_water"),
             "vav1_sham":_report("vav1_sham",loss=0.11)}
    with pytest.raises(ValueError,match="Sham reward"):
        paired_intervention_report(reports)


def test_unexecuted_treatment_not_misrepresented():
    reports={"pid_water":_report("pid_water"),
             "vav1_sham":_report("vav1_sham"),
             "vav1_reheat50":_report("vav1_reheat50",modified=0)}
    with pytest.raises(ValueError,match="No actuator"):
        paired_intervention_report(reports)
