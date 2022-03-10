from mimetypes import init
import gym
from gym import spaces
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from definitions import ROOT_DIR


class CookieJarsEnv(gym.Env):
    def __init__(self, split: str, initial_plate: float = 1e6, penalty_factor: float = 100) -> None:
        super().__init__()
        """
        :param split: 'train', 'val', or 'test' split. 
        """
        self.initial_plate = initial_plate
        self.penalty_factor = penalty_factor

        raw_df = pd.read_csv(ROOT_DIR / 'cookie_jars/data/stocks_data.csv')
        num_total = raw_df.shape[0]
        num_val = int(0.2 * num_total)
        num_test = num_val
        num_train = num_total - num_val - num_test
        if split == 'train':
            self.df = raw_df.iloc[:num_train, :]
        elif split == 'val':
            self.df = raw_df.iloc[num_train:(num_train + num_val), :]
        elif split == 'test':
            self.df = raw_df.iloc[(num_train + num_val):, :]
        else:
            raise ValueError(
                f"`split` argument must be one of (train, val, test). Offending arg: {split}"
            )

        assert self.df.columns[0] == 'time_id'
        self.episode_length = self.df.shape[0] - 1  # num steps in episode
        self.num_jars = self.df.shape[1] - 1

        # Action space: 1.0 represents 100% of cookie wealth (all cookies in plate and jars)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_jars,)
        )
        # Obs space: (num cookies in each jar... , bundle sizes..., num cookies on plate)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(2 * self.num_jars + 1,)
        )

        self.time_ind = None  # time index (diff from time_id in that it increments contiguously)
        self.jars = None
        self.bundle_sizes = None  # will be set in `reset`
        self.plate = None
        self.penalties = None
        self.done = None
    
    def reset(self) -> None:
        self.time_ind = 0
        self.jars = np.zeros((self.num_jars,))
        self.bundle_sizes = np.array(self.df.iloc[self.time_ind, 1:])
        self.plate = self.initial_plate
        self.penalties = 0
        self.done = False

        obs = np.concatenate((self.jars, self.bundle_sizes, [self.plate]))
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        
        if illegal action, do noop and give penalty (scaled by how much you went negative)
        """
        wealth_old = self.get_wealth()
        bundle_size_old = self.bundle_sizes
        temp_jars, temp_plate, penalty = self.dry_run_action(action)

        # action is legal if penalty is 0; if illegal, then don't apply action
        if penalty == 0:
            self.plate = temp_plate
            self.jars = temp_jars
        else:
            self.penalties += penalty

        # Now, traverse 1 time unit, growing/shrinking cookie jars
        self.time_ind += 1
        self.bundle_sizes = np.array(self.df.iloc[self.time_ind, 1:])
        self.jars *= self.bundle_sizes / bundle_size_old
        if self.time_ind == self.episode_length:
            self.done = True

        wealth_new = self.get_wealth()
        reward = wealth_new - wealth_old - penalty
        
        obs = np.concatenate((self.jars, self.bundle_sizes, [self.plate]))
        return obs, reward, self.done, {}

    def render(self, mode="human"):
        # return np.concatenate((self.jars, self.bundle_sizes, [self.plate]))
        return (
            f"total={self.plate + np.sum(self.jars) - self.penalties}, "
            f"plate={self.plate}, "
            f"sum_jars={np.sum(self.jars)}, "
            f"penalties={self.penalties}."
        )

    def dry_run_action(self, action: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        """
        wealth = self.get_wealth()
        temp_plate = self.plate - np.sum(action * wealth)
        temp_jars = self.jars + action * wealth
        
        # penalty indicates how badly your action turned you negative
        penalty = np.abs(temp_plate) * self.penalty_factor if temp_plate < 0 else 0
        neg_jars_mask = np.where(temp_jars < 0)
        penalty += np.sum(np.abs(temp_jars[neg_jars_mask]))
        
        return temp_jars, temp_plate, penalty
    
    def get_wealth(self) -> float:
        return self.plate + np.sum(self.jars)
