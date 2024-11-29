from stable_baselines3.common.env_checker import check_env

from environment_norm import AircraftEnv

env = AircraftEnv()

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

