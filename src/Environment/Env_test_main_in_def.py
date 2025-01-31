import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces

from pandapower.timeseries.data_sources.frame_data import DFData
import simbench as sb


class CustomEnv(gym.Env):
    def __init__(self, num_lines, num_switches, network, profiles, profiles_time_step, limit):
        self.observation = None
        self.count = 0
        self.profiles = None
        self.gen_data = None
        self.load_data = None
        self.truncated = False
        self.terminated = False
        self.ts = None
        self.info = dict()

        self.num_switches = num_switches
        self.num_lines = num_lines
        self.network = network
        self.profiles = profiles
        self.initial_net = None
        self.initial_net = self.network
        self.profiles_time_step = profiles_time_step
        self.limit = limit
        self.time_step = 0

        # spaces.MultiDiscrete from gym
        # self.action_space = spaces.MultiDiscrete([2]*(num_switches))
        self.action_space = spaces.MultiDiscrete([2] * network.switch.shape[0])
        self.discrete_space = spaces.MultiDiscrete([2] * network.switch.shape[0])  # Each switch can be 0 or 1

        ## line loadings, voltage, power losse -> 3
        # self.observation_space = spaces.Box(low=0, high=1, shape=(3), dtype =np.float32)
        ## line loadings -> 1
        num_generators = network.gen.shape[0]
        num_sgenerators = network.sgen.shape[0]
        num_loads = network.load.shape[0]
        #obs_space_size = num_lines + num_generators + num_sgenerators + num_loads
        obs_space_size = num_generators + num_sgenerators + num_loads
        self.continuous_space = spaces.Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)
        # self.observation_space= spaces.Box(low=0, high=10, shape=(num_switches + num_lines,), dtype =np.float64)
        # self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5349 + 953,), dtype=np.float64)
        self.observation_space = spaces.Dict({
            "discrete_switches": self.discrete_space,
            "continuous_grid": self.continuous_space,
        })
        ## MDP; state = obs
        # self.state = np.zeros(3)
        self.state = np.zeros(num_switches + num_lines)
        # self.state = np.zeros(5349 + 953)
        self.lineload_max = 0
        self.num_congested = 0
        self.lineload = np.zeros(num_lines)

        #self.load = profiles[("load", "p_mw")]
        #self.sgen = profiles[("sgen", "p_mw")]
        #self.gen = profiles[("gen", "p_mw")]

        gen_data_normalized = (self.profiles[('gen', 'p_mw')] - self.profiles[('gen', 'p_mw')].min()) / (
                self.profiles[('gen', 'p_mw')].max() - self.profiles[('gen', 'p_mw')].min())
        load_data_normalized = (self.profiles[('load', 'p_mw')] - self.profiles[('load', 'p_mw')].min()) / (
                self.profiles[('load', 'p_mw')].max() - self.profiles[('load', 'p_mw')].min())
        sgen_data_normalized = (self.profiles[('sgen', 'p_mw')] - self.profiles[('sgen', 'p_mw')].min()) / (
                self.profiles[('sgen', 'p_mw')].max() - self.profiles[('sgen', 'p_mw')].min())

        # Fill NaN with zeros
        self.gen_data_normalized = gen_data_normalized.fillna(0)
        self.load_data_normalized = load_data_normalized.fillna(0)
        self.sgen_data_normalized = sgen_data_normalized.fillna(0)


        #self.gen_data = gen_mw.values[self.profiles_time_step].astype(np.float32)
        #self.load_data = load_p.values[self.profiles_time_step].astype(np.float32)
        #self.sgen_data = sgen_p.values[self.profiles_time_step].astype(np.float32)

        self.gen_data = self.gen_data_normalized.values[self.profiles_time_step].astype(np.float32)
        self.load_data = self.load_data_normalized.values[self.profiles_time_step].astype(np.float32)
        self.sgen_data = self.sgen_data_normalized.values[self.profiles_time_step].astype(np.float32)


    def reset(self, seed=None, options=None, ts=None):
        if seed is not None:
            np.random.seed(seed)
        #self.state = np.random.rand(num_switches + num_lines).astype(np.float32)
        # self.state = np.random.rand(5349 + 953).astype(np.float32)

        self.network  = self.initial_net

        initial_discrete = self.network .switch['closed'].astype(int).values
        # Ensure that the concatenated values are in float32
        initial_continuous = np.concatenate([
            self.gen_data,
            self.sgen_data,
            self.load_data
        ], axis=0)

        self.observation = {
            "discrete_switches": initial_discrete,
            "continuous_grid": initial_continuous
        }

        return self.observation, self.info

    def step(self, action):
        # Update CB and DC states based on actions
        print('through into_step')

        self.terminated = False
        self.truncated = False
        self.count += 1

        # HARD CODED
        """
        if self.count == 1:
            for i in range(self.num_switches):
                self.network.switch.at[i, 'closed'] = True
        else:
            for i in range(self.network.switch.shape[0]):
                self.network.switch.at[i, 'closed'] = bool(action[i])
        """
        for i in range(self.network.switch.shape[0]):
            self.network.switch.at[i, 'closed'] = bool(action[i])
        # Update CB and DC states based on action
        #pp.diagnostic(self.network, report_style='detailed')

        #pp.diagnostic(self.network, report_style='compact')
        # load flow calculation
        # Extract line loading
        #time_step, lineload = self.ppcal_congestion(self.network, self.profiles, self.time_step, self.limit)
        try:
            time_step, lineload = self.ppcal_congestion(self.network, self.profiles, self.time_step, self.limit)
        except:
            self.terminated = True
            self.observation = self.observation
            return self.observation, 0, self.terminated, self.truncated, self.info
        self.lineload = lineload
        self.lineload_max = max(lineload)
        self.num_congested = np.sum(lineload > 1)
        self.avg_ll_congested = np.mean(lineload[lineload > 1])
        # Construct observation space
        self.state = np.concatenate((action, lineload))  # what should be in here?
        # self.state = np.concatenate((lineload[:5349 ],action[:953]))

        # Calculate reward
        reward = self.reward_calc(lineload, self.limit)
        # Placeholder for next state and done flag
        #self.observation = self.update_state()
        self.observation = self.observation
        # return self.state, reward, done, {}
        return self.observation, reward, self.terminated, self.truncated, self.info

    def ppcal_congestion(self, network, profiles, time_step, limit):
        pp.runpp(network, max_iteration=50, tolerance_mva=1e-4, enforce_q_lims=True, init='flat')
        lineload = 0.01 * network.res_line.loading_percent
        return time_step, lineload,

    def reward_calc(self, line_loadings_percent, limit):
        # line_loadings_percent = net.res_line.loading_percent
        num_line = len(line_loadings_percent)
        ll = line_loadings_percent
        ll_max = max(ll)
        # print("max line loading = ", ll_max)
        if ll_max >= 1:
            # If the maximum line loading is greater than or equal to 1, calculate the margin
            # Calculate the margin by subtracting the limit from the line loadings
            margin = ll - limit * np.ones(num_line)
            # Set any margins smaller than 1-limit to 0
            margin[margin < 1 - limit] = 0
            # Calculate the congestion metric 'u' as the sum of the margins (ignore nan values)
            u = np.nansum(margin)
            # Print the result for debugging purposes
            # print("ll_max>=1, u =", u)
        else:
            # Calculate the congestion metric 'u' when the maximum line loading is less than 1
            # This is done by subtracting the limit from the maximum line loading,
            # and ensuring the result is not negative
            u = max(ll_max - limit, 0)
            # Print the result for debugging purposes
            # print("ll_max<1, u =", u)

        reward = np.exp(-u)
        return reward




##grid model
def apply_absolute_values(network, absolute_values, case_or_time_step):
    """Apply absolute values to a network"""
    for element, parameter in absolute_values.keys():
        if absolute_values[(element, parameter)].shape[1]:
            network[element].loc[:, parameter] = absolute_values[(element, parameter)].loc[case_or_time_step]


# Simbench codes
simbench_codes = sb.collect_all_simbench_codes()
simbench_codes_lv = sb.collect_all_simbench_codes(lv_level="LV")

# Define complete data and grid codes
complete_data_codes = [f"1-complete_data-mixed-all-{i}-sw" for i in range(3)]
complete_grid_codes = [f"1-EHVHVMVLV-mixed-all-{i}-sw" for i in range(3)]

# Load a simbench network
network_code = "1-EHV-mixed--0-sw"
network = sb.get_simbench_net(network_code)
num_switches = max(network.switch.index) + 1
print("num_switches:", num_switches)

num_lines = max(network.line.index) + 1
print("print(num_lines)", num_lines)
# Get absolute values for the network
profiles = sb.get_absolute_values(network, profiles_instead_of_study_cases=True)

profiles_time_step = 0


# In Env.py
def update_profiles_time_step(new_time_step):
    global profiles_time_step
    profiles_time_step = new_time_step
    apply_absolute_values(network, profiles, profiles_time_step)

def update_reward_all_close(new_reward_all_close):
    global reward_all_close
    reward_all_close = new_reward_all_close



limit = 0.5
apply_absolute_values(network, profiles, profiles_time_step)

# Extract specific profiles
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]
sgen_p = profiles[("sgen", "p_mw")]
gen_mw = profiles[("gen", "p_mw")]
storage_mw = profiles[("storage", "p_mw")]

# Create and wrap the environment

from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import A2C

from stable_baselines3 import PPO  # or any other algorithm you used
import gym

env = CustomEnv(num_lines, num_switches, network, profiles, profiles_time_step, limit)
model =PPO.load(r"C:\Users\Jin\PycharmProjects\DRL_TopoOp\src\Environment\Agent.zip", env = env, custom_objects={"clip_range": 0.2, "lr_schedule": None})

# It will check your custom environment and output additional warnings if needed
#check_env(env)

# print("P")
lineload = env.lineload
lineload_max = env.lineload_max
####
# Convert lineload to a Series if it’s not already
lineload_series = pd.Series(lineload)

# Find all matches
matches = lineload_series[lineload_series == lineload_max]
#line_max = lineload[lineload == lineload_max].index[0]

# Check if there are any matches
if not matches.empty:
    line_max_indices = matches.index.tolist()  # Convert to list if needed
else:
    # Handle the case when no matching elements are found
    print("No match found for lineload_max in lineload.")
    line_max_indices = []  # Or handle this in a way that suits your program

# Now, line_max_indices contains all indices where lineload == lineload_max

#avg_ll_congested = env.avg_ll_congested
avg_ll_congested = np.mean(lineload[lineload > 1])
num_congested = env.num_congested

import time

def run_episodes():
    action_result = []
    previous_action = env.network.switch['closed'].astype(int).values
    action_result.append({'Previous Action': previous_action.tolist()})
    start_time = time.time()

    done = False

    obs, info = env.reset()
    action, state = model.predict(obs)
    # obs, reward, done, trunc, info = env.step(action)
    # print(f"Observation: {obs}")
    for i in range(env.network.switch.shape[0]):
        env.network.switch.at[i, 'closed'] = bool(action[i])
    try:
        pp.runpp(env.network, max_iteration=50, tolerance_mva=1e-4, enforce_q_lims=True, init='flat')
        done = True
    except:
        print("action cause pp conversion error")

    # print("Switches",env.network.switch)

    lineload_op = 0.01 * env.network.res_line.loading_percent
    lineload_max_op = np.nanmax(lineload_op)
    num_congested_op = np.sum((lineload_op > 1) & ~np.isnan(lineload_op))
    # avg_ll_congested_op = np.nanmean(lineload_op[lineload_op > 1])
    avg_ll_congested_op = np.nan_to_num(np.nanmean(lineload_op[lineload_op > 1]), nan=0)
    print(state)
    print(lineload_op)

    end_time = time.time()
    time_taken = end_time - start_time

    action_result.append({
        'Time': time_taken,
        'Action': action,
        # 'Line Loading': lineload_op,
        'Maximum line loading': lineload_max_op,
        'Number of congested lines': num_congested_op,
        'Average line loading of congested lines': avg_ll_congested_op,
        # 'Switch':env.network.switch
    })

    print("Action-Result:")
    for entry in action_result:
        print(entry)

    # Convert to a DataFrame and save as CSV
    df = pd.DataFrame(action_result)
    df.to_csv("action_results.csv", index=True)

    env.close()
    return "Optimization completed."
