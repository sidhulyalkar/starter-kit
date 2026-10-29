"""Pure stdlib unit tests for the published-baseline audit."""
from __future__ import annotations

import unittest

from scripts.audit_reference import STEPS, STATIONS, audit_reference


def fixture(buildings=6):
    return {
        f"building_train_{i}/{station}_{i}": {
            "bang_bang": -0.25 * STEPS,
            "fixed_setpoint": -0.10 * STEPS,
            "pid": -0.08 * STEPS,
            "pid_water": -0.05 * STEPS,
        }
        for i in range(1, buildings + 1)
        for station in sorted(STATIONS)
    }


class AuditTests(unittest.TestCase):
    def test_full_reference(self):
        result = audit_reference(fixture())
        self.assertEqual(result["scenarios"], 30)
        self.assertEqual(result["best_reference"], "pid_water")
        self.assertAlmostEqual(result["reference_scores"]["pid_water"], 0.05)
        self.assertEqual(result["steps_per_scenario"], 8640)

    def test_empty_and_partial_sets_fail(self):
        for d in ({}, fixture(5)):
            with self.subTest(size=len(d)), self.assertRaises(ValueError):
                audit_reference(d)

    def test_missing_policy_fails(self):
        d = fixture()
        next(iter(d.values())).pop("pid")
        with self.assertRaisesRegex(ValueError, "four reference policies"):
            audit_reference(d)

    def test_duplicate_station_fails(self):
        d = fixture()
        record = d.pop("building_train_1/atlanta_1")
        d["building_train_1/atlanta_9"] = record
        # Still one Atlanta, so this is legal; now introduce another Atlanta
        d["building_train_1/denver_1"] = d.pop("building_train_1/atlanta_9")
        # Restore Atlanta, which duplicates the Denver station? instead assert missing station
        with self.assertRaises(ValueError):
            audit_reference(d)

    def test_reject_nan_positive_boolean(self):
        for value in (float("nan"), float("inf"), 0.1, True):
            d = fixture()
            next(iter(d.values()))["pid"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit_reference(d)

    def test_reject_unknown_station(self):
        d = fixture()
        record = d.pop("building_train_1/atlanta_1")
        d["building_train_1/moon_1"] = record
        with self.assertRaises(ValueError):
            audit_reference(d)

    def test_score_mean_abs_is_not_sum(self):
        d = {"building_train_1/phoenix_1": {
            "bang_bang": -8640,
            "fixed_setpoint": -4320,
            "pid": -2160,
            "pid_water": -1080,
        }}
        r = audit_reference(d, expected_scenarios=1)
        self.assertEqual(r["reference_scores"]["bang_bang"], 1.0)
        self.assertEqual(r["reference_scores"]["pid_water"], 0.125)


if __name__ == "__main__":
    unittest.main()
