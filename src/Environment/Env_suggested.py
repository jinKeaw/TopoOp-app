import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from gymnasium import spaces
from pandapower.timeseries.data_sources.frame_data import DFData
import simbench as sb
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


# Define Custom Environment
class CustomEnv(gym.Env):
    def __init__(self, num_lines, num_switches, network, profiles, time_step, limit):
        self.num_switches = num_switches
        self.num_lines = num_lines
        self.network = network
        self.profiles = profiles
        self.time_step = time_step
        self.limit = limit

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([2] * num_switches)
        self.observation_space = spaces.Box(low=0, high=10, shape=(num_switches + num_lines,), dtype=np.float64)

        # Initialize environment variables
        self.state = np.zeros(num_switches + num_lines)
        self.lineload_max = 0
        self.num_congested = 0
        self.lineload = np.zeros(num_lines)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.rand(self.num_switches + self.num_lines).astype(np.float32)
        return self.state, dict()

    def step(self, action):
        # Apply actions to network
        for i in range(self.num_switches):
            self.network.switch.at[i, 'closed'] = True

        # Run load flow calculation and calculate congestion
        self.time_step, self.lineload = self.ppcal_congestion()
        self.lineload_max = max(self.lineload)
        self.num_congested = np.sum(self.lineload > 1)
        self.avg_ll_congested = np.mean(self.lineload[self.lineload > 1])

        # Update state and calculate reward
        self.state = np.concatenate((action, self.lineload))
        reward = self.reward_calc(self.lineload, self.limit)

        done = False
        truncated = False
        info = {}

        return self.state, reward, done, truncated, info

    def ppcal_congestion(self):
        pp.runpp(self.network)
        lineload = 0.01 * self.network.res_line.loading_percent
        return self.time_step, lineload

    def reward_calc(self, line_loadings_percent, limit):
        ll = line_loadings_percent
        ll_max = max(ll)

        if ll_max >= 1:
            margin = ll - limit * np.ones(len(ll))
            margin[margin < 1 - limit] = 0
            u = np.nansum(margin)
        else:
            u = max(ll_max - limit, 0)

        return np.exp(-u)


# Utility functions for grid modeling
def apply_absolute_values(network, absolute_values, case_or_time_step):
    for element, parameter in absolute_values.keys():
        if absolute_values[(element, parameter)].shape[1]:
            network[element].loc[:, parameter] = absolute_values[(element, parameter)].loc[case_or_time_step]


# Define simulation setup
simbench_codes = sb.collect_all_simbench_codes()
simbench_codes_lv = sb.collect_all_simbench_codes(lv_level="LV")

# Complete data codes
complete_data_codes = [f"1-complete_data-mixed-all-{i}-sw" for i in range(3)]
complete_grid_codes = [f"1-EHVHVMVLV-mixed-all-{i}-sw" for i in range(3)]

# Load network and apply profiles
network_code = "1-EHV-mixed--0-sw"
network = sb.get_simbench_net(network_code)
num_switches = network.switch.shape[0]
num_lines = network.line.shape[0]
profiles = sb.get_absolute_values(network, profiles_instead_of_study_cases=True)
profiles_time_step = 0
limit = 0.5

# Apply absolute values to the network
apply_absolute_values(network, profiles, profiles_time_step)

# Initialize environment
env = CustomEnv(num_lines, num_switches, network, profiles, profiles_time_step, limit)

# Check custom environment
check_env(env)

# Training section (commented out as placeholders)
# model = PPO("MlpPolicy", env)
# model.learn(total_timesteps=1000)

model =PPO.load(r"C:\Users\Jin\PycharmProjects\DRL_TopoOp\src\Environment\Agent.zip", env = env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())

env.close()