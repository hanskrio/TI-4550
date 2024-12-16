import torch
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from gym import Wrapper
from plotting import plot_boptest_results

# Constants
SECONDS_PER_DAY = 86_400  # Number of seconds in a day

# URL for the BOPTEST service.
url = 'http://localhost:5001/'

# Define the start times for the desired test period
# Peak Heat Day
peak_heating_test_start_day = 16  # Test period starts at Day 16
peak_heating_warmup_days = 7      # Warmup period is 7 days
peak_heating_start_day = peak_heating_test_start_day - peak_heating_warmup_days  # Day 9
peak_heating_start_time = peak_heating_start_day * SECONDS_PER_DAY

# Choose which period to simulate
start_time = peak_heating_start_time  # Or set to typical_heating_start_time if desired

# Define simulation periods
warmup_period = 7 * SECONDS_PER_DAY  # 1 week warmup
test_period = 14 * SECONDS_PER_DAY   # 2 weeks test
total_simulation_time = warmup_period + test_period

# Initialize the environment without specifying actions
env = BoptestGymEnv(
    url=url,
    actions=[],
    observations={
        'time': (0, 604800),
        'reaTZon_y': (280., 310.),
        'TDryBul': (265, 303),
        'HDirNor': (0, 862),
        'InternalGainsRad[1]': (0, 219),
        'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
        'LowerSetp[1]': (280., 310.),
        'UpperSetp[1]': (280., 310.)
    },
    predictive_period=24 * 3600,
    regressive_period=6 * 3600,
    max_episode_length=test_period,  # Two weeks test period
    warmup_period=warmup_period,     # One week warmup
    step_period=3600,
    start_time=start_time
    # No 'actions' parameter
)

# Define an empty action list to don't overwrite any input
env.actions = [] 

# Reset the environment to start the simulation
obs, _ = env.reset(options={'start_time': start_time})

# Initialize variables
done = False
rewards_list = []

# Loop through the simulation without specifying actions, allowing baseline control
while not done:
    # Step without a custom action to use baseline control
    obs, reward, terminated, truncated, info = env.step([])
    rewards_list.append(reward)
    done = terminated or truncated

# Plot the results for the benchmark controller
plot_boptest_results(env)