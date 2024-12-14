
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3 import DQN
import torch
import numpy as np

def run_rl_controller(start_time, warmup_period, test_period, step_period, url, model_path):
    """
    Runs the RL controller simulation.

    Parameters:
    - start_time: Simulation start time in seconds.
    - warmup_period: Warmup period in seconds.
    - test_period: Test period in seconds.
    - step_period: Simulation step period in seconds.
    - url: URL of the BOPTEST service.
    - model_path: Path to the trained RL model.

    Returns:
    - env_rl: The environment after the simulation.
    - kpis_rl: Dictionary of KPIs from the simulation.
    """

    # Check if MPS is available (for Mac M1 devices), otherwise fallback to CPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the environment
    env = BoptestGymEnv(
        url                  = url,
        actions              = ['oveHeaPumY_u'],
        observations         = {'time': (0, 604800),
                                'reaTZon_y': (280., 310.),
                                'TDryBul': (265, 303),
                                'HDirNor': (0, 862),
                                'InternalGainsRad[1]': (0, 219),
                                'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                                'LowerSetp[1]': (280., 310.),
                                'UpperSetp[1]': (280., 310.)},
        predictive_period    = 24 * 3600,  
        regressive_period    = 6 * 3600, 
        max_episode_length   = test_period,
        warmup_period        = warmup_period,
        step_period          = step_period,
        start_time           = start_time
    )

    # Normalize observations and discretize action space (if required)
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act=10)

    # Load the trained model
    model = DQN.load(model_path, env=env)

    # Initialize lists to store rewards
    rewards_list = []
    # Reset the environment with the specified start time
    obs, _ = env.reset(options={'start_time': start_time})
    done = False
    while not done:
        # Use the trained model to predict actions
        action, _ = model.predict(obs, deterministic=True)
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_list.append(reward)  # Collect rewards
        done = terminated or truncated

    # Obtain KPIs after the simulation
    kpis = env.get_kpis()
    print("Reinforcemenet Learning KPIs:")
    for key, value in kpis.items():
        print(f"{key}: {value}")

    # Return the environment, results DataFrame, and KPIs
    return env, kpis



    