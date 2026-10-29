"""Deterministic guards for truly frozen-context recursive forecasts."""
import numpy as np
import pytest

from smart_buildings.experiments.thermal_records import TransitionWriter
from smart_buildings.experiments.thermal_ridge import (
    FEATURE_NAMES, ThermalRidge, ZoneFeatureMap,
)
from smart_buildings.experiments.thermal_recursive import (
    frozen_context_rollout, evaluate_recursive,
)


def mapping():
    return ZoneFeatureMap(
        observation_dim=1, action_dim=1, discrete_dim=2,
        zone_items=((0,0,None,None,0,None,None,None),),
        ahu_observation_groups=((0,),), water_temp_idx=None,
        outside_name="", mean_kelvin=293.15, std_kelvin=10.0,
        clock_indices=(None,None,None,None), zone_names=("VAV_1",),
    )


def model(delta=0.1):
    return ThermalRidge(
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.r_[delta, np.zeros(len(FEATURE_NAMES))],
        alpha=1.0,
    )


def metadata():
    return {
        "observation_field_names": [
            "VAV_1_zone_air_temperature_sensor",
            "ahs_ahu_1_outside_air_temperature_sensor",
        ],
        "n_continuous_action":1,"n_discrete_action":2,
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {
                "sample_mean":293.15,"sample_std":10.0,
                "sample_variance":100.0,
            }
        },
        "action_normalization_metadata": {},
        "temp_indices":[0],"pressure_indices":[],
        "recirculation_indices":[],"water_indices":[],
        "zone_action_indices":{},
        "ahu_zone_assignments":{"ahu_1":["room_1"]},
        "floor_plan":None,"zone_map":None,
    }


def archive(path, building, drift=0.01):
    t=0.0
    with TransitionWriter(path, metadata(), {
        "scenario":f"{building}/denver_8","policy":"pid_water"
    }) as w:
        for k in range(36):
            next_t=t+drift
            w.write(k,[t,275.0],{
                "continuous_action":[0.5],
                "discrete_action":[1,1],
            },[next_t,275.0],-0.1)
            t=next_t


def test_rollout_uses_only_initial_observation_and_action():
    state=[0.0]
    result=frozen_context_rollout(model(),mapping(),state,[0.0],[1,1],12)
    assert result[0]==pytest.approx(294.35)
    assert state == [0.0], "Must not mutate the anchor observation"


@pytest.mark.parametrize("horizon", [0, -2, 1.5])
def test_invalid_horizons_rejected(horizon):
    with pytest.raises(ValueError):
        frozen_context_rollout(model(),mapping(),[0.0],[0.0],[1,1],horizon)


def test_evaluation_rejects_same_building(tmp_path):
    p=tmp_path/"data.jsonl.gz"
    archive(p,"building_0250")
    with pytest.raises(ValueError,match="leakage"):
        evaluate_recursive([p],[p])


def test_recursive_horizon_report(tmp_path):
    tr=tmp_path/"train.jsonl.gz"
    ev=tmp_path/"eval.jsonl.gz"
    archive(tr,"building_0250")
    archive(ev,"building_0410",drift=0.015)
    report=evaluate_recursive([tr],[ev],alpha=10,
                              horizons=(1,3,12),anchor_stride=3)
    assert report["eval_buildings"]==["building_0410"]
    result=report["per_archive"][0]["by_horizon"]
    assert result["1"]["available"]
    assert result["3"]["available"]
    assert result["12"]["available"]
    assert result["1"]["anchors"] > result["12"]["anchors"]
    assert result["12"]["frozen_action_ridge"]["mae_kelvin"] >= 0
    assert "NOT counterfactual" in report["interpretation"]
