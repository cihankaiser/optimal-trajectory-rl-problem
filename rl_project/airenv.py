import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AirEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,goal_x,goal_y,goal_h):
        super(AirEnv).__init__()
        # define constraints for observation space
        self.max_x = 90*111000
        self.max_y = 90*111000
        self.min_y = 0
        self.min_x = 0
        self.max_h = 12800
        self.min_h = 0
        self.max_v = 300
        self.min_v = 70  # stall speed
        self.max_m = 70000
        self.min_m = 32000
        self.max_psi = 6.28 # in radians
        self.min_psi = 0

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_h = goal_h

        # define constraints for action space
        self.max_gamma = 1 # 15 degrees in radians
        self.min_gamma = -1 # -15 degrees in radians and normalized by 15 degrees (0.262 radians)
        self.max_mu = 1  # 45 degrees in radians
        self.min_mu = -1  # -45 degrees in radians
        self.max_d = 1
        self.min_d = -1

        self.low_state = np.array([self.min_x,self.min_y,self.min_h, self.min_v, self.min_m, self.min_psi], dtype=np.float32)
        self.high_state = np.array([self.max_x,self.max_y,self.max_h, self.max_v, self.max_m, self.max_psi], dtype=np.float32)
        self.low_action = np.array([self.min_gamma, self.min_mu, self.min_d], dtype=np.float32)
        self.high_action = np.array([self.max_gamma, self.max_mu, self.max_d],  dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.low_action, high=self.high_action, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, shape=(6,), dtype=np.float32
        )
    def step(self, action):
        x = self.state[0]
        y = self.state[1]
        h = self.state[2]
        v = self.state[3]
        psi = self.state[5]
        m = self.state[4]
        gamma = min(max(action[0],self.min_gamma),self.max_gamma)*0.262
        mu = min(max(action[1],self.min_mu),self.max_mu)*0.785
        d = 0.5*(min(max(action[2],self.min_d),self.max_d)) + 0.5

        # dynamics
        # wind model
        fi = x/111000
        lamb = y/111000/np.cos(fi)
        fi =  np.deg2rad(fi)
        lamb =  np.deg2rad(lamb)

        cx0 = -21.151 
        cx1 = 10.0039 
        cx2 = 1.1081 
        cx3 = -0.5239 
        cx8=-0.0001
        cx4 = -0.1297 
        cx5 = -0.006 
        cx6=0.0073
        cx7 = 0.0066

        cy0 = -65.3035
        cy1 = 17.6148
        cy2 = 1.0855
        cy3 = -0.7001 
        cy8=-0.000227
        cy4 = -0.5508
        cy5 = -0.003
        cy6=0.0241
        cy7 = 0.0064

        wx = cx0+cx1*lamb+cx2*fi+cx3*fi*lamb+cx4*lamb**2+cx5*fi**2 + cx6*fi*lamb**2+cx7*lamb*fi**2+cx8*lamb**2*fi**2
        wy = cy0+cy1*lamb+cy2*fi+cy3*fi*lamb+cy4*lamb**2+cy5*fi**2 +cy6*fi*lamb**2+cy7*lamb*fi**2+cy8*lamb**2*fi**2

        # aircraft model
        # maximum thrust model
        ctcr = 0.95
        ctc1 = 146590
        ctc2 = 53872
        ctc3 = 3.0453*10**(-11)
        thrmax = ctcr*ctc1*(1-3.28*h/ctc2+ctc3*(3.28*h)**2)

        # air density model
        rho_0 = 1.225; 
        rho = rho_0*(1-(2.2257*10**-5)*h)**4.2586

        # lift coefficient model
        g = 9.80665
        S = 124.65    
        cl = 2*m*g/(rho*S*np.cos(mu)*v**2)

        # drag coefficient model
        cd0 = 0.025452
        k = 0.035815
        cd=cd0+k*cl**2

        # fuel consumption model
        cf1 = 0.70057
        cf2 = 1068.1
        eta = (cf1/60000)*(1+1.943*v/cf2)
        cfcr = 0.92958
        f = d*thrmax*eta*cfcr

        # state equations
        x += v*np.cos(psi)*np.cos(gamma) + wx
        y += v*np.sin(psi)*np.cos(gamma) + wy
        h += v*np.sin(gamma)
        v += thrmax*d/m - 0.5*rho*v**2*S*cd/m - g*np.sin(gamma)
        psi += cl*S*rho*v*np.sin(mu)/(2*m)/np.cos(gamma)
        m -= f

                # this might be changed because of the reward function
        # approx might be wrong, and we should use the distance function provided
        if x > self.max_x:
            terminated = True
        if x < self.min_x:
            terminated = True
        if y > self.max_y:
            terminated = True
        if y < self.min_y:
            terminated = True
        if h > self.max_h:
            terminated = True
        if h < self.min_h:
            terminated = True
        if v > self.max_v:
            terminated = True
        if v < self.min_v:
            terminated = True
        if m > self.max_m:
            terminated = True
        if m < self.min_m:
            terminated = True
        if psi > self.max_psi:
            terminated = True
        if psi < self.min_psi:
            terminated = True
        
        # this might be changed because of the reward function
        # approx might be wrong, and we should use the distance function provided
        if ( x >= self.goal_x and y >= self.goal_y and h >= self.goal_h ):
            terminated = True
        else:
            terminated = False
        reward = 0
        distance = np.sqrt((x-self.goal_x)**2+(y-self.goal_y)**2+(h-self.goal_h)**2)
        if distance < 1000:
            reward = 100000.0

        reward -= (0.05 + f) + distance*0.001

        if x > self.max_x:
            x = self.max_x
        if x < self.min_x:
            x = self.min_x
        if y > self.max_y:
            y = self.max_y
        if y < self.min_y:
            y = self.min_y
        if h > self.max_h:
            h = self.max_h
        if h < self.min_h:
            h = self.min_h
        if v > self.max_v:
            v = self.max_v
        if v < self.min_v:
            v = self.min_v
        if m > self.max_m:
            m = self.max_m
        if m < self.min_m:
            m = self.min_m
        if psi > self.max_psi:
            psi = self.max_psi
        if psi < self.min_psi:
            psi = self.min_psi
        

        self.state = np.array([x,y,h,v,m,psi], dtype=np.float32)


        
        
        return self.state, reward, terminated, False, {}
    def reset(self, seed = None, options = None):

        self.state = np.array([4440000,425154,8000,210,68000,0], dtype=np.float32)


        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        ...

    def close(self):
        ...