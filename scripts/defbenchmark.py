import numpy as np
import pandas as pd
import seaborn as sns
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from gym import Wrapper

def run_benchmark_controller(start_time, warmup_period, test_period, step_period, url):
    """
    Runs the benchmark controller simulation without overriding control signals.

    Parameters:
    - start_time: Simulation start time in seconds.
    - warmup_period: Warmup period in seconds.
    - test_period: Test period in seconds.
    - step_period: Simulation step period in seconds.
    - url: URL of the BOPTEST service.

    Returns:
    - env: The environment after the simulation.
    - df_benchmark: DataFrame containing the simulation results.
    - kpis_benchmark: Dictionary of KPIs from the simulation.
    """
    # Initialize the environment without actions
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
        max_episode_length=test_period,
        warmup_period=warmup_period,
        step_period=step_period,
        start_time=start_time
    )
    
    # Define an empty action list to don't overwrite any input
    env.actions = [] 

    # Reset the environment with the specified start time
    obs, _ = env.reset(options={'start_time': start_time})
    
    # Initialize variables
    done = False
    rewards_list = []
    
    # Loop through the simulation without specifying actions
    while not done:
        obs, reward, terminated, truncated, info = env.step([])
        rewards_list.append(reward)
        done = terminated or truncated
    
    # Obtain KPIs after the simulation
    kpis_benchmark = env.get_kpis()
    print("Benchmark KPIs:")
    for key, value in kpis_benchmark.items():   
        print(f"{key}: {value}")
    
    return env, kpis_benchmark