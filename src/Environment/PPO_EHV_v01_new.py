import gymnasium
import numpy as np
import os
import datetime
import time
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.spaces import MultiDiscrete, Box
import gymnasium.spaces as spaces
import simbench as sb
import pandapower as pp
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Define environment class
class ENV_EHV_v01(gymnasium.Env):
    def __init__(self,
                 simbench_code="1-EHV-mixed--0-sw",
                 case_study='bc',
                 is_train=True,
                 ):
        super().__init__()

        self.simbench_code = simbench_code
        self.net = self.load_simbench_net()
        self.initial_net = None
        self.is_train = is_train
        self.time_step = 0
        self.observation = None
        self.count = 0
        self.profiles = None
        self.gen_data = None
        self.load_data = None
        self.truncated = False
        self.terminated = False
        self.ts = None
        self.info = dict()
        self.sgen_data = None
        self.case_study = case_study
        self.test_data_length = None
        self.train_data_length = None
        self.override_timestep = None

        self.action_space, self.observation_space = self.create_act_obs_space()

        self.set_study_case(case_study, self.is_train, load_all=True)

        _ = self.reset()

        # Define constants and parameters
        self.gamma = 0.99  # Discount factor
        self.rho_max = 1.0  # Maximum acceptable load rate

    def set_study_case(self, case_study, is_train, load_all=True):

        if load_all:
            self.case_study = case_study

            loadcases = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=False)

            self.net = self.apply_absolute_values(self.net, loadcases, self.case_study)

            self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)

            # Normalize to the range [0, 1]
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

            # Split into train and test sets (80% train, 20% test)
            gen_train, gen_test = train_test_split(self.gen_data_normalized, test_size=0.2, shuffle=False)
            load_train, load_test = train_test_split(self.load_data_normalized, test_size=0.2, shuffle=False)
            sgen_train, sgen_test = train_test_split(self.sgen_data_normalized, test_size=0.2, shuffle=False)

            self.test_data_length = gen_test.shape[0]
            self.train_data_length = gen_train.shape[0]

            # Based on the train flag, use the appropriate data
            if is_train:
                self.gen_data = gen_train
                self.load_data = load_train
                self.sgen_data = sgen_train
            else:
                # print("i am test")
                self.gen_data = gen_test
                self.load_data = load_test
                self.sgen_data = sgen_test

            self.initial_net = self.net

        return

    def create_act_obs_space(self):
        # Define the action space: ON/OFF for each CB and DC
        action_space = spaces.MultiDiscrete([2] * self.net.switch.shape[0])

        discrete_space = MultiDiscrete([2] * self.net.switch.shape[0])  # Each switch can be 0 or 1

        num_lines = 0  # self.net.line.shape[0]
        num_generators = self.net.gen.shape[0]
        num_sgenerators = self.net.sgen.shape[0]
        num_loads = self.net.load.shape[0]
        obs_space_size = num_lines + num_generators + num_sgenerators + num_loads

        # Define the continuous space (4 grid elements between 0 and 1)  Sgen, gen, load, ext_grid
        continuous_space = Box(low=0.0, high=1.0, shape=(obs_space_size,), dtype=np.float32)  ## ERROR
        # Combine the spaces into a single observation space
        observation_space = spaces.Dict({
            "discrete_switches": discrete_space,
            "continuous_grid": continuous_space,
        })

        return action_space, observation_space

    def load_simbench_net(self):
        net = sb.get_simbench_net(self.simbench_code)
        return net

    def set_to_all_data(self):
        self.gen_data = self.gen_data_normalized
        self.load_data = self.load_data_normalized
        self.sgen_data = self.sgen_data_normalized
        return

    def apply_absolute_values(self, net, absolute_values_dict, case_or_time_step):
        for elm_param in absolute_values_dict.keys():
            if absolute_values_dict[elm_param].shape[1]:
                elm = elm_param[0]
                param = elm_param[1]
                net[elm].loc[:, param] = absolute_values_dict[elm_param].loc[case_or_time_step]
        return net

    def reset(self, options=None, seed=None, ts=None):

        # If timestep is not provided, use a random timestep based on available data
        if ts is None:
            self.time_step = np.random.randint(0, self.gen_data.shape[0])
            relative_index = self.gen_data.index.values[self.time_step]
        else:
            self.time_step = ts
            # print('This')

        self.net = self.initial_net

        initial_discrete = self.net.switch['closed'].astype(int).values
        # Ensure that the concatenated values are in float32
        initial_continuous = np.concatenate([
            self.gen_data.values[self.time_step].astype(np.float32),
            self.sgen_data.values[self.time_step].astype(np.float32),
            self.load_data.values[self.time_step].astype(np.float32)
        ], axis=0)

        self.observation = {
            "discrete_switches": initial_discrete,
            "continuous_grid": initial_continuous
        }

        # Run load flow calculations
        try:
            pp.runpp(self.net)
        except:
            return self.observation, self.info

        self.net = self.apply_absolute_values(self.net, self.profiles, relative_index)

        return self.observation, self.info

    def step(self, action):

        self.terminated = False
        self.truncated = False
        self.count += 1

        # Update CB and DC states based on action
        for i in range(self.net.switch.shape[0]):
            self.net.switch.at[i, 'closed'] = bool(action[i])

        if self.count == 5:
            self.truncated = True
            self.terminated = True

        # Run load flow calculations
        try:
            pp.runpp(self.net)
        except:
            self.terminated = True
            self.observation = self.update_state()
            return self.observation, 0, self.terminated, self.truncated, self.info

        # Extract load rates (rho), voltages, and power losses from pandapower results
        self.net.res_line['loading_percent'] = self.net.res_line['loading_percent'].fillna(0)

        # Compute P_j_t
        P_j_t = np.array([line['loading_percent'] / 100 for _, line in self.net.res_line.iterrows()])

        # Calculate rewards
        R_congestion_t = self.calculate_congestion_reward(P_j_t)

        # Calculate combined reward with adaptive weights
        R_t = R_congestion_t

        # Placeholder for next state and done flag
        self.observation = self.update_state()

        return self.observation, R_t, self.terminated, self.truncated, self.info

    def get_data_length(self):
        return self.test_data_length, self.train_data_length

    def calculate_congestion_reward(self, rho):
        u_t = np.sum(rho - 0.5)
        # print(f"Congestion: {u_t}")  # Debugging line
        R_congestion = np.exp(-u_t) / (1 + np.exp(-u_t))
        # print(f"Congestion Reward: {R_congestion}")  # Debugging line
        return R_congestion

    def update_state(self):
        # Update the state based on the current time step
        self.time_step += 1
        # Update the state of the discrete switches and continuous grid
        if self.time_step >= self.gen_data.shape[0]:
            self.time_step = 0

        initial_discrete = self.net.switch['closed'].astype(int).values
        initial_continuous = np.concatenate([
            self.gen_data.values[self.time_step].astype(np.float32),
            self.sgen_data.values[self.time_step].astype(np.float32),
            self.load_data.values[self.time_step].astype(np.float32)
        ], axis=0)

        relative_index = self.gen_data.index.values[self.time_step]

        # update the network
        self.net = self.apply_absolute_values(self.net, self.profiles, relative_index)

        # Return the updated state as a dictionary
        return {
            "discrete_switches": initial_discrete,
            "continuous_grid": initial_continuous
        }


# Define custom callbacks
class TimeStepLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        time_elapsed = time.time() - self.start_time
        self.logger.record("time_elapsed", time_elapsed)
        self.logger.record("total_timesteps", self.num_timesteps)
        return True

def main():
    # Create a monitored environment and wrap it in a vectorized environment
    monitored_env = Monitor(ENV_EHV_v01(is_train=True))
    env = DummyVecEnv([lambda: monitored_env])

    # Define the logging directory
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Instantiate the PPO model and start learning
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, n_epochs=1, n_steps=5, batch_size=5)
    model.learn(total_timesteps=10_000, tb_log_name="PPO_new",progress_bar=True, reset_num_timesteps=False, callback=TimeStepLoggingCallback())

    # Save the model
    model.save("PPO_EHV_v01_new")

if __name__ == "__main__":
    main()
