"""Offline *simulator-internal* audit of recurring single-zone thermal jumps.

This is research instrumentation, NOT a submission policy: it intentionally
reads simulator-internal thermal capacity, realized actuator settings and
source powers that must NEVER become runtime policy inputs.

Example:
  python -m scripts.diagnose_zone_physics \
    --building building_train_2 --weather denver_8 --zone VAV_1 \
    --steps 288 --threshold-k 2 --results-dir smart_buildings/results/g3_physics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.run_adaptive_experiments import POLICY_MAP
from smart_buildings.evaluate import (
    create_building_meta_data, get_policy_action, initialize_policy,
    load_hybrid_environment, validate_action,
)
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker


def _finite(x, label: str) -> float:
    val = float(x)
    if not math.isfinite(val):
        raise ValueError(f"Nonfinite internal diagnostic: {label}")
    return val


def energy_budget_report(records: list[dict], threshold_k: float = 2.0) -> dict:
    """Rank observed jumps; do not mislabel source-power estimates as exact.

    Thermal diffusion, air-side implicit coupling and convection can change
    temperatures besides the logged local heat source. The source-power
    approximation is not an energy-conservation residual.
    """
    if not records or threshold_k <= 0 or not math.isfinite(threshold_k):
        raise ValueError("Require records and finite positive shock threshold")
    rows = []
    for row in records:
        d = _finite(row["delta_kelvin"], "zone delta")
        if abs(d) >= threshold_k:
            rows.append(row)
    rows.sort(key=lambda row: abs(row["delta_kelvin"]), reverse=True)
    return {
        "steps": len(records),
        "shock_threshold_kelvin": threshold_k,
        "shock_count": len(rows),
        "shock_fraction": len(rows)/len(records),
        "top_12_shocks": rows[:12],
        "shock_steps": [row["step"] for row in sorted(rows, key=lambda r:r["step"])],
        "max_absolute_delta_kelvin": max(abs(r["delta_kelvin"]) for r in records),
        "interpretation": (
            "Internal simulator fields are research diagnostics ONLY, never "
            "permitted controller inputs. Thermal-source delta estimates omit "
            "diffusion, implicit air coupling, wall exchange and convection. "
            "No causal mechanism established by temporal correlation alone."),
    }


def diagnose(building: str, weather: str, zone: str, policy_name: str,
             steps: int, threshold_k: float, output_root: Path) -> dict:
    if steps < 1 or policy_name not in POLICY_MAP or threshold_k <= 0:
        raise ValueError("Invalid scenario/policy/step/threshold parameters")
    config = Path("smart_buildings/training_scenarios")/building/weather/"config.gin"
    if not config.is_file():
        raise FileNotFoundError(config)

    env = load_hybrid_environment(config)
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, config.parent)
    policy = initialize_policy(POLICY_MAP[policy_name], meta, env.action_spec())
    sensor_name = f"{zone}_zone_air_temperature_sensor"
    names = meta["observation_field_names"]
    if sensor_name not in names:
        raise ValueError(f"Unknown zone sensor: {sensor_name}")
    sensor_idx = names.index(sensor_name)
    n = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
    temp_mean, temp_std = _finite(n["sample_mean"], "mean"), _finite(n["sample_std"], "std")
    sim = env.building.simulator
    hvac = sim.hvac
    if zone not in hvac.vavs:
        raise ValueError(f"VAV absent from simulator: {zone}")
    vav = hvac.vavs[zone]
    rooms = hvac.get_zones_for_vav(zone)
    if not rooms:
        raise ValueError(f"No rooms assigned to VAV: {zone}")
    if len(rooms) != 1:
        raise ValueError("Diagnostic presently supports one room per VAV")
    room = rooms[0]
    cap = _finite(sim.building.zone_air_heat_capacity(room), "zone thermal capacity")
    if cap <= 0:
        raise ValueError("Nonpositive zone heat capacity")
    cv_area_m2 = (sim.building.cv_size_cm/100.0)**2
    room_area = len(sim.building.room_dict[room])*cv_area_m2
    action_idxs = meta["zone_action_indices"][zone]
    tracker.reset()
    observations = tracker.last_obs
    out = []
    for step in range(steps):
        before = _finite(observations[sensor_idx]*temp_std+temp_mean, "temperature_t")
        cmd = validate_action(get_policy_action(policy, observations), env.action_spec())
        observations, _reward, terminated, truncated, _info = tracker.step(cmd)
        after = _finite(observations[sensor_idx]*temp_std+temp_mean, "temperature_tp1")
        ahu = vav.air_handler
        ahu_id = ahu.device_id()
        # IMPORTANT: AHU supply temperature sensor in this implementation
        # reports the *setpoint*. Economizer mixing may differ from this
        # value, so no exact coil energy is inferred from it.
        thermal_source_w = _finite(sim.building.get_zone_thermal_energy_rate(room),
                                   "zone applied source power")
        damper_idx = action_idxs.get("supply_air_damper_percentage_command")
        reheat_idx = action_idxs.get("reheat_valve_percentage_command")
        record = {
            "step": step,
            "minute": 5*step,
            "zone": zone,
            "room": room,
            "ahu": ahu_id,
            "temp_t_kelvin": before,
            "temp_tp1_kelvin": after,
            "delta_kelvin": after-before,
            "native_room_area_m2": room_area,
            "internal_zone_air_heat_capacity_j_per_kelvin": cap,
            "source_power_w_reported_by_simulator": thermal_source_w,
            "source_only_delta_k_approx": thermal_source_w*sim.time_step_sec/cap,
            "reported_ahu_supply_setpoint_kelvin": _finite(
                ahu.supply_air_temperature_setpoint, "supply setpoint"),
            "reported_hot_water_supply_kelvin": _finite(
                hvac.hot_water_system.supply_water_temperature_sensor,
                "water supply"),
            "realized_zone_air_flow_m3_s": _finite(
                vav.flow_rate_demand, "zone airflow"),
            "realized_damper_fraction": _finite(vav.damper_setting, "damper"),
            "realized_reheat_fraction": _finite(vav.reheat_valve_setting, "reheat"),
            "commanded_damper_normalized": (
                float(cmd["continuous_action"][damper_idx]) if damper_idx is not None else None),
            "commanded_reheat_normalized": (
                float(cmd["continuous_action"][reheat_idx]) if reheat_idx is not None else None),
        }
        out.append(record)
        if terminated or truncated:
            break
    summary = energy_budget_report(out, threshold_k)
    report = {
        "scenario": f"{building}/{weather}",
        "policy": policy_name,
        "zone": zone,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "room_floor_area_m2": room_area,
        "room_heat_capacity_j_per_kelvin": cap,
        "data_scope": "offline internal simulator physics diagnostics",
        "summary": summary,
        "trace": out,
    }
    target = output_root/building/weather/policy_name
    target.mkdir(parents=True, exist_ok=True)
    (target/f"{zone}_physics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "trace"},
                     indent=2, allow_nan=False))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--building", default="building_train_2")
    p.add_argument("--weather", default="denver_8")
    p.add_argument("--policy", default="pid_water", choices=sorted(POLICY_MAP))
    p.add_argument("--zone", default="VAV_1")
    p.add_argument("--steps", default=288, type=int)
    p.add_argument("--threshold-k", default=2.0, type=float)
    p.add_argument("--results-dir", type=Path,
                   default=Path("smart_buildings/results/g3_physics"))
    args=p.parse_args()
    diagnose(args.building,args.weather,args.zone,args.policy,args.steps,
             args.threshold_k,args.results_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
