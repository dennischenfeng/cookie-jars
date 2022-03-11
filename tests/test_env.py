"""
Unit tests for env.py
"""

import pytest
from cookie_jars.env import CookieJarsEnv
import numpy as np


def test_env_obs_and_rew():
    env = CookieJarsEnv('train')
    env.reset()
    assert env.time_ind == 0

    action = np.zeros(31)
    action[-1] = 1.0
    obs, rew, done, info = env.step(action)
    jars = obs[:env.num_jars]
    bundle_sizes = obs[env.num_jars:-1]
    plate = obs[-1]
    assert env.time_ind == 1
    assert np.all(jars == 0)
    np.testing.assert_allclose(bundle_sizes[:3], [82.80, 40.44, 40.83], atol=0.01)
    assert plate == 1e6
    assert env.get_wealth() == 1e6
    assert rew == 0

    action = np.zeros(31)
    wealth = env.get_wealth()
    action[0] = 0.0001
    action[2] = 0.0002
    action[4] = 0.0010
    action[-1] = 0.9987
    obs, rew, done, info = env.step(action)
    jars = obs[:env.num_jars]
    bundle_sizes = obs[env.num_jars:-1]
    plate = obs[-1]
    assert env.time_ind == 2
    assert jars[0] == pytest.approx(100 * 83.88 / 82.80, abs=0.01)
    assert jars[2] == pytest.approx(200 * 41.23 / 40.83, abs=0.01)
    assert jars[4] == pytest.approx(1000 * 16.21 / 15.74, abs=0.01)
    np.testing.assert_allclose(bundle_sizes[:3], [83.88, 39.19, 41.23], atol=0.01)
    assert plate == 1e6 - 100 - 200 - 1000
    assert env.get_wealth() > 1e6
    assert rew > 0

    action = np.zeros(31)
    wealth = env.get_wealth()
    action[0] = 0.0002
    action[2] = 0.0001
    action[4] = 0.0010
    action[-1] = 0.9987
    obs, rew, done, info = env.step(action)
    jars = obs[:env.num_jars]
    bundle_sizes = obs[env.num_jars:-1]
    plate = obs[-1]
    assert env.time_ind == 3
    assert jars[0] == pytest.approx(
        0.0002 * wealth * 83.32 / 83.88, 
        abs=0.01,
    )
    assert jars[2] == pytest.approx(
        0.0001 * wealth * 41.26 / 41.23, 
        abs=0.01,
    )
    assert plate == 0.9987 * wealth

    # # illegal action
    # action = np.zeros(31)
    # wealth = env.get_wealth()
    # action[0] = 0 / wealth
    # action[2] = -1000 / wealth
    # obs, rew, done, info = env.step(action)
    # jars = obs[:env.num_jars]
    # bundle_sizes = obs[env.num_jars:-1]
    # plate = obs[-1]
    # assert jars[2] == pytest.approx(400, rel=0.05)
    # assert plate == 1e6 - 50 - 2 * 200 - 2 * 1000  # unchanged from last time step
    # assert rew == pytest.approx(-600, rel=0.10)


def test_env_done():
    env = CookieJarsEnv('train')
    env.reset()

    action = np.zeros(31)

    for i in range(env.episode_length - 1):
        obs, rew, done, info = env.step(action)
        assert not done 
    
    obs, rew, done, info = env.step(action)
    assert done


def test_env_bad_split():
    with pytest.raises(ValueError):
        _ = CookieJarsEnv('trian')  # notice the typo

