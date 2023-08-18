from airenv import AirEnv
import numpy as np
goal_x = 40*111000
goal_y = 32*111000*np.cos(np.deg2rad(40))
goal_h = 8000
env = AirEnv(goal_x,goal_y,goal_h)
episodes = 50
actions = list(([0,0,0],[0.1,0.1,0.1],[0.2,0.2,0.2],[0.2,-0.2,-0.2],
                   [0.3,0.3,0.3],[0.4,0.4,0.4],[0.5,0.5,0.5],[1,1,1]))
for action in actions:
    obs = env.reset()
    obs, rewards, dones,truncated, info, = env.step(np.array(action))
    print(obs)

    