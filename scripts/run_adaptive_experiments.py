"""Register experimental policies without changing upstream evaluate.py.

Examples:
  python -m scripts.run_adaptive_experiments --policies pid_water adaptive \
    --buildings building_train_1 --weathers atlanta_2 --steps 288
  python -m scripts.run_adaptive_experiments --policies pid_water adaptive \
    --results-dir smart_buildings/results/qualification --steps 8640
"""
from smart_buildings.evaluate import POLICY_MAP, main
from smart_buildings.experiments.adaptive_policy import (
    AdaptivePIDPolicy, SensorOnlyPolicy, OutlierOnlyPolicy, ForecastOnlyPolicy,
)

POLICY_MAP.update({
    "adaptive": AdaptivePIDPolicy,
    "adaptive_sensor": SensorOnlyPolicy,
    "adaptive_outlier": OutlierOnlyPolicy,
    "adaptive_forecast": ForecastOnlyPolicy,
})

if __name__ == "__main__":
    main()
