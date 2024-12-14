import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns


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
    plt.show()



def plot_comparison_boptest_results(df_res_list, labels, save_directory=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    sns.set_style("dark")
    sns.set_context("paper")
    palette = sns.color_palette("muted")

    plt.figure(figsize=(12, 10))

    # Define colors for each controller
    colors = [palette[3], palette[4]]  # Assign distinct colors
    colors2 = [palette[1], palette[-1]]

    # Subplot 1: Zone Temperature and Setpoints
    ax1 = plt.subplot(3, 1, 1)
    for i, (df_res, label) in enumerate(zip(df_res_list, labels)):
        ax1.plot(df_res['time_days'], df_res['reaTZon_y'] - 273.15,
                 label=f'Zone Temp {label}', color=colors[i], linestyle='-', linewidth=1.5)
    # Plot setpoints (assuming they are the same)
    df_res_setpoints = df_res_list[0]  # Use the first dataset for setpoints
    ax1.plot(df_res_setpoints['time_days'], df_res_setpoints['reaTSetHea_y'] - 273.15,
             label='Heating Setpoint', color=palette[7], linestyle='--', linewidth=1.5)
    ax1.plot(df_res_setpoints['time_days'], df_res_setpoints['reaTSetCoo_y'] - 273.15,
             label='Cooling Setpoint', color=palette[7], linestyle='-.', linewidth=1.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.set_title('Zone Temperature and Setpoints')

    # Subplot 2: Heat Pump Modulation Signal
    ax2 = plt.subplot(3, 1, 2)
    for i, (df_res, label) in enumerate(zip(df_res_list, labels)):
        ax2.plot(df_res['time_days'], df_res['oveHeaPumY_u'],
                 label=f'Heat Pump Signal {label}', color=colors2[i], linestyle='-', linewidth=1.5)
    ax2.set_ylabel('Control Signal (-)')
    ax2.legend()
    ax2.set_title('Heat Pump Modulation Signal')

    # Subplot 3: Ambient Conditions
    ax3 = plt.subplot(3, 1, 3)
    df_res_ambient = df_res_list[0]  # Use the first dataset for ambient conditions
    ax3.plot(df_res_ambient['time_days'], df_res_ambient['weaSta_reaWeaTDryBul_y'] - 273.15,
             label='Outdoor Temp', color=palette[0], linewidth=1.5)
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper left')

    # Add secondary y-axis for solar radiation
    axt = ax3.twinx()
    axt.plot(df_res_ambient['time_days'], df_res_ambient['weaSta_reaWeaHDirNor_y'],
             label='Solar Radiation', color=palette[8], linewidth=1.5)
    axt.set_ylabel('Solar Radiation (W/m²)')
    axt.legend(loc='upper right')

    ax3.set_title('Ambient Conditions')

    plt.tight_layout()

    # Save the figure if save_directory is provided
    if save_directory:
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)
        # Create a filename using the labels
        filename = f"comparison_{labels[0].replace(' ', '_')}_{labels[1].replace(' ', '_')}.png"
        save_path = os.path.join(save_directory, filename)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()