from stable_baselines3.common.env_checker import check_env
from airenv import AirEnv
import numpy as np
goalx1 = 40*111000
goaly1 = 32*111000*np.cos(np.deg2rad(40))
goalh1 = 8000
env = AirEnv(goalx1,goaly1,goalh1)
# It will check your custom environment and output additional warnings if needed
check_env(env)