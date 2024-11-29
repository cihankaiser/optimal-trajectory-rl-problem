import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from environment_norm import AircraftEnv
from stable_baselines3.common.env_util import make_vec_env
env = AircraftEnv()

env.reset()
log_dir = "C:/Users/cihan/OneDrive/Dokumente"
os.makedirs(log_dir, exist_ok=True)

models_dir = "C:/Users/cihan/OneDrive/Dokumente/models"
os.makedirs(models_dir, exist_ok=True)

TIMESTEPS = 20000
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, n_steps=4096, ent_coef=0.001)

iters = 0
j = 1
while TIMESTEPS*iters < 10000000:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    iters += 1