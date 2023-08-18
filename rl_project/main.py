import gym
from stable_baselines3 import PPO
import os
import numpy as np
import time
from airenv import AirEnv

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
	



goal_x = 40*111000
goal_y = 32*111000*np.cos(np.deg2rad(40))
goal_h = 8000
env = AirEnv(goal_x,goal_y,goal_h)

env.reset()

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logdir)
TIMESTEPS = 10000

for i in range(100):
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*i}")