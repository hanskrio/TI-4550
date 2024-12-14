import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from defbenchmark import run_benchmark_controller
from defrl import run_rl_controller
from plotting import plot_comparison_boptest_results

# Constants
SECONDS_PER_DAY = 86_400  # Number of seconds in a day

# URL for the BOPTEST service.
url = 'http://localhost:5001/'

def get_simulation_results(env, points=None):

    """
    Retrieve results from a BOPTEST simulation environment.

    Parameters:
    - env: The environment used in the simulation.

    Returns:
    - df_res: DataFrame containing the simulation results.
    """
    # Unwrap the environment to access base attributes
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    # Get the URL and other attributes
    url = base_env.url
    start_time = base_env.start_time
    try:
        sim_time = base_env.sim_time  # Current simulation time
    except AttributeError:
        sim_time = start_time + base_env.max_episode_length

    # Define the points to retrieve if not provided
    if points is None:
        points = [
            'reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 'oveHeaPumY_u',
            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'
        ]

    # Retrieve the results from the BOPTEST API
    args = {
        'point_names': points,
        'start_time': start_time,
        'final_time': sim_time
    }
    res = requests.put(f'{url}/results', json=args).json()['payload']
    # Print the lengths of each array in 'res'
    for key, value in res.items():
        print(f"Key: {key}, Length: {len(value)}")
    df_res = pd.DataFrame(data=res)

    # Check if data retrieval was successful
    if df_res.empty:
        print("No data retrieved from the simulation.")
        return df_res

    # Adjust time to start from zero and convert to hours and days
    df_res['time'] = df_res['time'] - df_res['time'].iloc[0]
    df_res['time_hours'] = df_res['time'] / 3600
    df_res['time_days'] = df_res['time_hours'] / 24

    return df_res

# Define the start times for the desired test period
# Peak Heat Day
peak_heating_test_start_day = 16  # Test period starts at Day 16
peak_heating_warmup_days = 7      # Warmup period is 7 days
peak_heating_start_day = peak_heating_test_start_day
peak_heating_start_time = peak_heating_start_day * SECONDS_PER_DAY

# Typical Heat Day
typical_heating_test_start_day = 108  # Test period starts at Day 108
typical_heating_warmup_days = 7       # Warmup period is 7 days
typical_heating_start_day = typical_heating_test_start_day
typical_heating_start_time = typical_heating_start_day * SECONDS_PER_DAY


# Choose which period to simulate
start_time = typical_heating_start_time  # Set period (peak/typical)

# Define simulation periods
warmup_period = 7 * SECONDS_PER_DAY  # 1 week warmup
test_period = 14 * SECONDS_PER_DAY   # 2 weeks test
total_simulation_time = warmup_period + test_period

# Path to trained RL model
model_path = "/Users/hanskrio/Desktop/NTNU/Prosjektoppgave/code/RL_energy/DQN_summer_constant.zip"
url = 'http://localhost:5001/'

# Retrieve results for both simulations using the environments
env_benchmark, kpis_benchmark = run_benchmark_controller(start_time=start_time, warmup_period=warmup_period, test_period=test_period, step_period=3600, url=url)
df_benchmark = get_simulation_results(env_benchmark)

# Reset and start the next controller
reset_response = requests.put(f'{url}/initialize', data={'start_time': start_time})
env, kpis_rl = run_rl_controller(start_time=start_time, warmup_period=warmup_period, test_period=test_period, step_period=3600, url=url, model_path=model_path)
df_rl = get_simulation_results(env)


save_directory = "/Users/hanskrio/Desktop/NTNU/Prosjektoppgave/code/RL_energy/pics"
# Plot the comparison
plot_comparison_boptest_results([df_rl, df_benchmark], ['RL controller', 'Benchmark controller'],save_directory=save_directory)