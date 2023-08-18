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
models_dir = "models/1691318545"
model_path = f"{models_dir}/80000.zip"
model = PPO.load(model_path, env=env)

episodes = 10


for i in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action,_ = model.predict(obs)
		obs, rewards, done,truncated, info, = env.step(action)