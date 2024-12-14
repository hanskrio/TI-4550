from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3 import DQN
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


# Check if MPS is available (for Mac M1 devices), otherwise fallback to CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# URL for the BOPTEST service.
url = 'http://localhost:5001/' 

# Modify the environment to simulate for two weeks (14 days)
env = BoptestGymEnv(
    url                  = url,
    actions              = ['oveHeaPumY_u'],
    observations         = {'time':(0,604800),
                            'reaTZon_y':(280.,310.),
                            'TDryBul':(265,303),
                            'HDirNor':(0,862),
                            'InternalGainsRad[1]':(0,219),
                            'PriceElectricPowerHighlyDynamic':(-0.4,0.4),
                            'LowerSetp[1]':(280.,310.),
                            'UpperSetp[1]':(280.,310.)},
    predictive_period    = 24 * 3600,  # Predict over two weeks
    regressive_period    = 6 * 3600, 
    max_episode_length   = 7 * 24 * 3600,  # Simulate for 7 days
    warmup_period        = 24 * 3600,  # 1-day warmup
    step_period          = 3600  # 1-hour steps
)

# Normalize observations and discretize action space (if required)
env = NormalizedObservationWrapper(env)
env = DiscretizedActionWrapper(env, n_bins_act=10)

# Assuming the model is already trained and in memory
# You can instantiate the model again here if necessary
# If the model was saved, you could also load it like this:
model = DQN.load("/Users/hanskrio/Desktop/NTNU/Prosjektoppgave/code/RL_energy/DDQN_200k.zip", env=env)

test_periods = [
    (432000, 1036800),        # January 5 - January 12
#    (5184000, 5788800),       # March 1 - March 8
#    (27648000, 28108800)      # November 16 - November 23
]

def plot_boptest_results(env, points=None):
    """
    Retrieve and plot results from the BOPTEST simulation.

    Parameters:
    - env: The environment used in the simulation.
    - points: List of variables to retrieve and plot.
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
    df_res = pd.DataFrame(data=res)

    # Check if data retrieval was successful
    if df_res.empty:
        print("No data retrieved from the simulation.")
        return

    # Adjust time to start from zero and convert to hours and days
    df_res['time'] = df_res['time'] - df_res['time'].iloc[0]
    df_res['time_hours'] = df_res['time'] / 3600
    df_res['time_days'] = df_res['time_hours'] / 24

    # Plot the results
    sns.set_style("dark")
    sns.set_context("paper")
    palette = sns.color_palette("muted") # Styling

    plt.figure(figsize=(12, 8))

    # Plot zone temperature and setpoints
    plt.subplot(3, 1, 1)
    plt.plot(df_res['time_days'], df_res['reaTZon_y'] - 273.15, label='Zone Temp', color=palette[3], linewidth=1.5)
    plt.plot(df_res['time_days'], df_res['reaTSetHea_y'] - 273.15, label='Heating Setpoint', color=palette[7], linestyle='--', linewidth=1.5)
    plt.plot(df_res['time_days'], df_res['reaTSetCoo_y'] - 273.15, label='Cooling Setpoint', color=palette[7], linestyle='-.', linewidth=1.5)
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title('Zone Temperature and Setpoints')

    # Plot control signal
    plt.subplot(3, 1, 2)
    plt.plot(df_res['time_days'], df_res['oveHeaPumY_u'], label='Heat Pump Signal', color=palette[1], linewidth=1.5)
    plt.ylabel('Control Signal (-)')
    plt.legend()
    plt.title('Heat Pump Modulation Signal')

    # Plot outdoor conditions
    plt.subplot(3, 1, 3)
    plt.plot(df_res['time_days'], df_res['weaSta_reaWeaTDryBul_y'] - 273.15, label='Outdoor Temp', color=palette[0], linewidth=1.5)
    plt.ylabel('Temperature (°C)')
    plt.xlabel('Time (days)')
    plt.legend(loc='upper left')

    # Add secondary y-axis for solar radiation
    axt = plt.gca().twinx()
    axt.plot(df_res['time_days'], df_res['weaSta_reaWeaHDirNor_y'], label='Solar Radiation', color=palette[8], linewidth=1.5)
    axt.set_ylabel('Solar Radiation (W/m²)')
    axt.legend(loc='upper right')

    plt.title('Ambient Conditions')

    plt.tight_layout()
    plt.savefig(f'/Users/hanskrio/Desktop/NTNU/Prosjektoppgave/code/RL_energy/pics/{start_time}')

# Initialize a list to store KPIs for each period
all_kpis = []

for idx, (start_time, end_time) in enumerate(test_periods):
    print(f"\nTesting on period {idx + 1}: Start Time = {start_time}, End Time = {end_time}")

    # Reset the environment with the specified start_time
    obs, _ = env.reset(options={'start_time': start_time})

    done = False
    rewards_list = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_list.append(reward)
        done = terminated or truncated

    # Obtain KPIs after the episode
    kpis = env.get_kpis()
    all_kpis.append(kpis)

    # Print KPIs for the current period
    print(f"Evaluation KPIs for period starting at {start_time}:")
    for key, value in kpis.items():
        print(f"{key}: {value}")

    plot_boptest_results(env)
