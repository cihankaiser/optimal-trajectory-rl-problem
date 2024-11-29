import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AircraftEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32),
                                       high=np.array([1, 1, 1], dtype=np.float32),
                                       dtype=np.float32)
        self.min_states = np.array([35.0, -5.0, 5000.0, 100.0, -2*np.pi, 10000.0, 0.0], dtype=np.float32)
        self.max_states = np.array([60.0, 35.0, 15000.0, 400.0, 2*np.pi, 69000.0, 2400000.0], dtype=np.float32)
        self.min_actions = np.array([-0.0261799, -0.5259, 0.2], dtype=np.float32)
        self.max_actions = np.array([0.0261799, 0.5259, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_states, 
                                            high=self.max_states,
                                            dtype=np.float32)
        self.state = None
        self.time = None
        
    def step(self, action):
        delta = 0.5 * (action[2] + 1) * (self.max_actions[2] - self.min_actions[2]) + self.min_actions[2]
        gamma = 0.5 * (action[0] + 1) * (self.max_actions[0] - self.min_actions[0]) + self.min_actions[0]
        mu = 0.5 * (action[1] + 1) * (self.max_actions[1] - self.min_actions[1]) + self.min_actions[1]

        lat, lon, h, v, psi, m, d = self.state

        x, y = self.lla2ned(lat, lon, [40, 5])
        xd, yd = self.lla2ned(40, 32, [40, 5])
        hd = 8000.0  # destination for flight 1
        dt = 1  # time step in seconds

        # Calculate wind once to avoid redundancy
        wx, wy = self.wind(lat, lon)

        # implement the dynamics model
        xdot = v * np.cos(psi) * np.cos(gamma) + wx
        ydot = v * np.sin(psi) * np.cos(gamma) + wy
        hdot = v * np.sin(gamma)
        vdot = (self.Thrust(h) * delta - self.drag(v, h, mu, m)) / m - 9.8065 * np.sin(gamma)
        psidot = self.lift_coefficient(v, h, mu, m) * 124.65 * self.rho(h) * v * np.sin(mu) / (2 * m) / np.cos(gamma)
        mdot = -self.fuel_consumption(v, delta, h)

        # update states
        x += xdot * dt
        y += ydot * dt
        h += hdot * dt
        v += vdot * dt
        psi += psidot * dt
        m += mdot * dt
        d = np.sqrt((x - xd) ** 2 + (y - yd) ** 2 + (h - hd) ** 2)
        lat, lon = self.ned2lla(x, y, [40, 5])

        # calculate reward
        reward = -self.calculate_cost(delta, h, v, d)
        terminated = False

        if self.reached_destination(d):
            reward = 1e5
            terminated = True
            print("Reached destination")

        if lat < 39.0 or lat > 41.0 or lon < 4.0 or lon > 33.0 or h < 7000.0 or h > 13000.0:
            reward = -1e5
            terminated = True

        self.time += dt
        if self.time > 15000:
            terminated = True

        self.state = np.array([lat, lon, h, v, psi, m, d], dtype=np.float32)

        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.state = np.array([40.0, 5.0, 8000.0, 210.0, 0.0, 68000.0, 2302443.2], dtype=np.float32)
        self.time = 0.0
        return self.state, {}

    def render(self):
        pass

    def close(self):
        pass
    
    def calculate_cost(self, delta, h, v, d):
        return d ** 2 / 1e14 + 0.05 + self.fuel_consumption(v, delta, h)
    
    def wind(self, lat, lon):
        cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8 = -21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001
        cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8 = -65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227
        
        Wx = cx0 + cx1 * lon + cx2 * lat + cx3 * lon * lat + \
             cx4 * lon ** 2 + cx5 * lat ** 2 + cx6 * lon ** 2 * lat + \
             cx7 * lon * lat ** 2 + cx8 * lon ** 2 * lat ** 2
        Wy = cy0 + cy1 * lon + cy2 * lat + cy3 * lon * lat + \
             cy4 * lon ** 2 + cy5 * lat ** 2 + cy6 * lon
        return Wx, Wy
    
    def Thrust(self, h):
        CTc1, CTc2, CTc3, CTcr = 146590.0, 53872.0, 3.0453e-11, 0.95
        Thrust_max = CTcr * CTc1 * (1 - (3.28 * h / CTc2) + CTc3 * (3.28 * h) ** 2)
        return Thrust_max
    
    def fuel_consumption(self, v, delta, h):
        Cfc1, Cfc2, Cfcr = 0.70057, 1068.1, 0.92958
        eta = Cfc1 / 60000 * (1 + 1.943 * v / Cfc2)
        f = delta * self.Thrust(h) * eta * Cfcr
        return f
    
    def rho(self, h):
        rho0 = 1.225
        rho = rho0 * (1 - (2.25577e-5) * h) ** 4.2586
        return rho
    
    def drag(self, v, h, mu, m):
        S = 124.65
        Cd0 = 0.025452
        k = 0.035815
        Cl = 2 * m * 9.8065 / (self.rho(h) * S * v ** 2 * np.cos(mu))
        Cd = Cd0 + k * Cl ** 2
        D = 0.5 * self.rho(h) * S * Cd * v ** 2
        return D
    
    def lift_coefficient(self, v, h, mu, m):
        S = 124.65
        Cl = 2 * m * 9.8065 / (self.rho(h) * S * v ** 2 * np.cos(mu))
        return Cl
    
    def lla2ned(self, lat, lon, ref):
        lat0, lon0 = ref
        Re = 6371000
        x = Re * (lat - lat0) * np.pi / 180
        y = Re * np.cos(lat0 * np.pi / 180) * (lon - lon0) * np.pi / 180
        return x, y
    
    def ned2lla(self, x, y, ref):
        lat0, lon0 = ref
        Re = 6371000
        lat = lat0 + x * 180 / np.pi / Re
        lon = lon0 + y * 180 / (np.pi * Re * np.cos(lat0 * np.pi / 180))
        return lat, lon
    
    def reached_destination(self, d):
        return d < 500