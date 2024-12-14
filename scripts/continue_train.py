from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3 import DQN
import torch
import matplotlib.pyplot as plt
import numpy as np

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# url for the BOPTEST service. 
url = 'http://localhost:5001/' 

# Decide the state-action space of your test case
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
        predictive_period    = 24*3600, 
        regressive_period    = 6*3600, 
        max_episode_length   = 7*24*3600,
        warmup_period        = 24*3600,
        step_period          = 3600)

# Normalize observations and discretize action space
env = NormalizedObservationWrapper(env)
env = DiscretizedActionWrapper(env,n_bins_act=10)

model = DQN.load("trained_dqn_model_100000_steps", env=env, device=device)
model.load_replay_buffer("trained_dqn_model_100000_steps.pkl")
# Main training loop
model.learn(total_timesteps=100000)
model.save("trained_dqn_model_200000_steps")
model.save_replay_buffer("replay_buffer_200.pkl")

# Loop for one episode of experience (7 days)
done = False
obs, _ = env.reset()
while not done:
  action, _ = model.predict(obs, deterministic=True) 
  obs,reward,terminated,truncated,info = env.step(action)
  done = (terminated or truncated)

# Obtain KPIs for evaluation
kpis = env.get_kpis()
print("Evaluation KPIs:")
for key, value in kpis.items():
    print(f"{key}: {value}")