import pandapower as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from pandapower.timeseries.data_sources.frame_data import DFData
import simbench as sb

# Simbench codes
simbench_codes = sb.collect_all_simbench_codes()
simbench_codes_lv = sb.collect_all_simbench_codes(lv_level="LV")

# Define complete data and grid codes
complete_data_codes = [f"1-complete_data-mixed-all-{i}-sw" for i in range(3)]
complete_grid_codes = [f"1-EHVHVMVLV-mixed-all-{i}-sw" for i in range(3)]

# Load a simbench network
network_code = "1-EHV-mixed--0-sw"
network = sb.get_simbench_net(network_code)

def apply_absolute_values(network, absolute_values, case_or_time_step):
    """Apply absolute values to a network"""
    for element, parameter in absolute_values.keys():
        if absolute_values[(element, parameter)].shape[1]:
            network[element].loc[:, parameter] = absolute_values[(element, parameter)].loc[case_or_time_step]

# Get absolute values for the network
profiles = sb.get_absolute_values(network, profiles_instead_of_study_cases=True)

# Extract specific profiles
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]
sgen_p = profiles[("sgen", "p_mw")]
gen_mw = profiles[("gen", "p_mw")]
storage_mw = profiles[("storage", "p_mw")]


def reward_calc(line_loadings_percent, limit):
    #line_loadings_percent = net.res_line.loading_percent
    num_line = len(line_loadings_percent)
    ll = line_loadings_percent
    ll_max = max(ll)
    print("max line loading = ", ll_max)
    if ll_max >= 1:
        # If the maximum line loading is greater than or equal to 1, calculate the margin
        # Calculate the margin by subtracting the limit from the line loadings
        margin = ll - limit * np.ones(num_line)
        # Set any margins smaller than 1-limit to 0
        margin[margin < 1 - limit] = 0
        # Calculate the congestion metric 'u' as the sum of the margins (ignore nan values)
        u = np.nansum(margin)
        # Print the result for debugging purposes
        print("ll_max>=1, u =", u)
    else:
        # Calculate the congestion metric 'u' when the maximum line loading is less than 1
        # This is done by subtracting the limit from the maximum line loading,
        # and ensuring the result is not negative
        u = max(ll_max - limit, 0)
        # Print the result for debugging purposes
        print("ll_max<1, u =", u)

    reward = np.exp(-u)
    return reward, u

"""
time_steps = range(24)
results = pd.DataFrame([], index=time_steps, columns=["max line loading", "min_vm_pu", "max_vm_pu"])

    # Reward (congestion) calculation

for time in time_steps:
    apply_absolute_values(net, profiles, time)

    pp.runpp(net)
    ll = net.res_line.loading_percent

    results.loc[time, "max line loading"] = ll.max()
#    results.loc[time_step, "min_vm_pu"] = net.res_bus.vm_pu.min()
#    results.loc[time_step, "max_vm_pu"] = net.res_bus.vm_pu.max()

    # Reward (congestion) calculation
    reward=reward_calc(ll, limit =0.5)
    print("Hour:", time, "reward =", reward)
    print("Hour:", time, ", loading_percent:")
    print(ll)
"""

"""
time_step = 200
apply_absolute_values(network, profiles, time_step)
pp.runpp(network)
ll = 0.01 * network.res_line.loading_percent
"""

def step(network, profiles, time_step, limit):
    apply_absolute_values(network, profiles, time_step)
    pp.runpp(network)
    lineload = 0.01 * network.res_line.loading_percent
    reward, margin_u = reward_calc(lineload, limit)
    return  time_step,lineload, margin_u, reward

time_step1, ll1, margin_u1, reward1= step(network, profiles, 200, 0.5)
time_step2, ll2, margin_u2, reward2= step(network, profiles, 300, 0.5)

"""
results = pd.DataFrame([], index=time_steps, columns=["max line loading", "min_vm_pu", "max_vm_pu"])
results.loc[time, "max line loading"] = ll.max()
#    results.loc[time_step, "min_vm_pu"] = net.res_bus.vm_pu.min()
#    results.loc[time_step, "max_vm_pu"] = net.res_bus.vm_pu.max()
"""

num_cbs = network.switch
# Reward (congestion) calculation
#reward1, margin_u1=reward_calc(ll1, limit =0.5)
#print(ll1)
print("time step1:", time_step1, "margin, u =", margin_u1)
print("time step1:", time_step1, "reward =", reward1)


#reward2, margin_u2=reward_calc(ll2, limit =0.5)
#print(ll2)
print("time step2:", time_step2, "margin, u =", margin_u2)
print("time step2:", time_step2, "reward =", reward2)

#print(ll2)

#num_switch = max(network.switch.index)+1
#print("num_cbs:",num_cbs)


