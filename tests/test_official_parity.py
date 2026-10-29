"""Fail closed on fast-vs-official evaluation differences."""
import pytest

from scripts.check_official_parity import verify_parity


def _pair():
    return (
        {"policy":"pid_water","scenario":"building_train_1/atlanta_2",
         "total_steps":288,"total_reward":-4.2},
        {"total_steps":288,"total_reward":-4.2},
    )


def test_identical_reward_is_valid():
    a,b=_pair()
    r=verify_parity(a,b)
    assert r["status"]=="matched"
    assert r["absolute_reward_difference"]==0


def test_reward_mismatch_fails():
    a,b=_pair()
    b["total_reward"]=-4.199
    with pytest.raises(ValueError,match="reward mismatch"):
        verify_parity(a,b)


def test_episode_length_mismatch_fails():
    a,b=_pair()
    b["total_steps"]=12
    with pytest.raises(ValueError,match="horizon mismatch"):
        verify_parity(a,b)


def test_nonfinite_and_invalid_tolerance_fails():
    a,b=_pair()
    b["total_reward"]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        verify_parity(a,b)
    a,b=_pair()
    with pytest.raises(ValueError,match="tolerance"):
        verify_parity(a,b,-0.1)
