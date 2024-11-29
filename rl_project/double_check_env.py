from environment_norm import AircraftEnv
import numpy as np

env = AircraftEnv()
episodes = 50
print(env.reset())
for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		
		obs, reward, done, truncated,info = env.step(random_action)

		print('reward',reward)
		print("obs",obs)

		if np.isnan(obs).any() or np.isnan(reward):
			print("NAN")
			break
