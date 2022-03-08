import gym
from gym import spaces
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from definitions import ROOT_DIR

NUM_JARS = 30
INVALID_ACTION_PENALTY_FACTOR = 100


class CookieJarsEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        """
        obs space
        action space
        obs - ()
        state (includes datetime)
        """
        # Action space: 1.0 represents 100% of cookie wealth (all cookies in plate and jars)
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(NUM_JARS,))
        # Obs space: (num bundles in each jar... , bundle sizes..., num cookies on plate)
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(2 * NUM_JARS + 1,)
        )

        self.df = pd.read_csv(ROOT_DIR / 'cookie_jars/data/stocks_data.csv')

        self.time_ind = None  # time index (diff from time_id in that it increments contiguously)
        self.jars = None
        self.bundle_sizes = None  # will be set in `reset`
        self.plate = None
        self.penalties = None
        self.done = None
    
    def reset(self) -> None:
        self.time_ind = 0
        self.jars = np.zeros((NUM_JARS,), dtype=int)
        self.bundle_sizes = np.array(self.df.iloc[self.time_ind, 1:])
        self.plate = 0
        self.penalties = 0
        self.done = False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        if illegal action, do noop and give penalty (scaled by how much you went negative)
        """
        wealth_before = self.get_wealth()
        temp_jars, temp_plate, penalty = self.dry_run_action()
        
        # action is legal if penalty is 0; if illegal, then don't apply action
        if penalty == 0:
            self.plate = temp_plate
            self.jars = temp_jars
        else:
            self.penalties += penalty

        # Now, fast-forward 1 time unit        
        # TODO: update bundle_sizes and update `done`
        self.time_ind += 1
        self.bundle_sizes = np.array(self.df.iloc[self.time_ind, 1:])
        if self.time_ind == self.df.shape[0] - 1:
            self.done = True

        wealth_after = self.get_wealth()
        reward = wealth_before - wealth_after - penalty
        
        obs = np.concatenate((self.jars, self.bundle_sizes, [self.plate]))
        return obs, reward, self.done, {}

    def render(self, mode="human"):
        return np.concatenate((self.jars, self.bundle_sizes, [self.plate]))

    def dry_run_action(self, action: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        """
        temp_plate = self.plate - np.sum(action)
        temp_jars = self.jars + action
        
        # penalty indicates how badly your action turned you negative
        penalty = np.abs(temp_plate) * INVALID_ACTION_PENALTY_FACTOR if temp_plate < 0 else 0
        neg_jars_mask = np.where(temp_jars < 0)
        penalty += np.sum(np.abs(temp_jars[neg_jars_mask]))
        
        return temp_jars, temp_plate, penalty
    
    def get_wealth(self) -> float:
        return self.plate + np.sum(self.jars * self.bundle_sizes)