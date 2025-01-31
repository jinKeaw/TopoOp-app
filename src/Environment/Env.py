import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#import gym
#from gym import spaces
import gymnasium as gym
from gymnasium import spaces

from pandapower.timeseries.data_sources.frame_data import DFData
import simbench as sb

class CustomEnv(gym.Env):
    def __init__(self,num_lines, num_switches, network, profiles, time_step, limit):
        self.num_switches = num_switches
        self.num_lines = num_lines
        self.network=network
        self.profiles= profiles
        self.time_step=time_step
        self.limit=limit
        #spaces.MultiDiscrete from gym
        #self.action_space = spaces.MultiDiscrete([2]*(num_switches))
        self.action_space = spaces.MultiDiscrete([2] * (num_switches))
        ## line loadings, voltage, power losse -> 3
        #self.observation_space = spaces.Box(low=0, high=1, shape=(3), dtype =np.float32)
        ## line loadings -> 1
        self.observation_space= spaces.Box(low=0, high=10, shape=(num_switches + num_lines,), dtype =np.float64)
        #self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5349 + 953,), dtype=np.float64)

        ## MDP; state = obs
        #self.state = np.zeros(3)
        self.state = np.zeros(num_switches + num_lines)
        #self.state = np.zeros(5349 + 953)
        self.lineload_max = 0
        self.num_congested = 0
        self.lineload = np.zeros(num_lines)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.rand(num_switches + num_lines).astype(np.float32)
        #self.state = np.random.rand(5349 + 953).astype(np.float32)
        return self.state, dict()

    def step(self, action):
        # Update CB and DC states based on actions
        print('through into_step')

        #HARD CODED
        for i in range(self.num_switches):
            #self.network.switch.at[i, 'closed'] = bool(action[i])
            self.network.switch.at[i, 'closed'] = True


        #load flow calculation
        # Extract line loading
        time_step, lineload=self.ppcal_congestion(self.network,self.profiles,self.time_step,self.limit)
        self.lineload = lineload
        self.lineload_max = max(lineload)
        self.num_congested = np.sum(lineload > 1)
        self.avg_ll_congested = np.mean(lineload[lineload>1])
        # Construct observation space
        self.state = np.concatenate((action, lineload)) #what should be in here?
        #self.state = np.concatenate((lineload[:5349 ],action[:953]))

        # Calculate reward
        reward = self.reward_calc(lineload, self.limit)

        done = False
        truncated = False
        info = {}
        #return self.state, reward, done, {}
        return self.state, reward, done, truncated, info
    def ppcal_congestion(self,network, profiles, time_step, limit):
        pp.runpp(network)
        lineload = 0.01 * network.res_line.loading_percent
        return time_step, lineload,

    def reward_calc(self, line_loadings_percent, limit):
        # line_loadings_percent = net.res_line.loading_percent
        num_line = len(line_loadings_percent)
        ll = line_loadings_percent
        ll_max = max(ll)
        #print("max line loading = ", ll_max)
        if ll_max >= 1:
            # If the maximum line loading is greater than or equal to 1, calculate the margin
            # Calculate the margin by subtracting the limit from the line loadings
            margin = ll - limit * np.ones(num_line)
            # Set any margins smaller than 1-limit to 0
            margin[margin < 1 - limit] = 0
            # Calculate the congestion metric 'u' as the sum of the margins (ignore nan values)
            u = np.nansum(margin)
            # Print the result for debugging purposes
            #print("ll_max>=1, u =", u)
        else:
            # Calculate the congestion metric 'u' when the maximum line loading is less than 1
            # This is done by subtracting the limit from the maximum line loading,
            # and ensuring the result is not negative
            u = max(ll_max - limit, 0)
            # Print the result for debugging purposes
            #print("ll_max<1, u =", u)

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
num_switches = max(network.switch.index)+1
print("num_switches:",num_switches)

num_lines = max(network.line.index)+1
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

limit =0.5
apply_absolute_values(network, profiles, profiles_time_step)

# Extract specific profiles
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]
sgen_p = profiles[("sgen", "p_mw")]
gen_mw = profiles[("gen", "p_mw")]
storage_mw = profiles[("storage", "p_mw")]

# Create and wrap the environment

from stable_baselines3.common.env_checker import check_env
#from stable_baselines3 import A2C

from stable_baselines3 import PPO  # or any other algorithm you used
import gym

env = CustomEnv(num_lines,num_switches, network, profiles, profiles_time_step, limit)
#model =PPO.load(r"C:\Users\Jin\PycharmProjects\DRL_TopoOp\src\Environment\Agent.zip", env = env)


#model = A2C("MlpPolicy", env).learn(total_timesteps=1000)

# It will check your custom environment and output additional warnings if needed
check_env(env)


#print("P")
lineload = env.lineload
lineload_max = env.lineload_max
line_max = lineload[lineload == lineload_max].index[0]
avg_ll_congested = env.avg_ll_congested
num_congested = env.num_congested


