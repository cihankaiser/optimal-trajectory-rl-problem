import gymnasium as gym
import numpy as np
from gymnasium import spaces

class AircraftEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(AircraftEnv).__init__()
        # Define action and observation space
        # actions: [gamma,mu,delta]
        self.action_space = spaces.Box(low = np.array([-1, -1, -1],dtype=np.float32), high = np.array([1, 1, 1],dtype=np.float32),dtype=np.float32,shape=(3,))
        # observations: [latitude,longitude,altitude,airspeed,heading,mass, distance to goal] (normalized between min and max values)
        self.min_states = np.array([35.0,-5.0,5000.0,100.0,-2*np.pi,10000.0,0.0],dtype=np.float32)
        self.max_states = np.array([60.0,35.0,15000.0,400.0,2*np.pi,69000.0,2400000.0],dtype=np.float32)
        self.min_actions = np.array([-0.0261799,-0.5259,0.2],dtype=np.float32)
        self.max_actions = np.array([0.0261799,0.5259,1],dtype=np.float32)
        self.observation_space = spaces.Box(low = self.min_states, high = self.max_states,dtype=np.float32,shape=(7,))
        self.state = None
        self.time = None
        
    def step(self, action):
        # scale the state and action back to the original range
        delta = 0.5*(action[2]+1)*(self.max_actions[2]-self.min_actions[2])+self.min_actions[2]
        gamma = 0.5*(action[0]+1)*(self.max_actions[0]-self.min_actions[0])+self.min_actions[0]
        mu = 0.5*(action[1]+1)*(self.max_actions[1]-self.min_actions[1])+self.min_actions[1]
        
        lat,lon,h,v,psi,m,d = self.state

        x,y = self.lla2ned(lat,lon,[40,5])
        xd,yd = self.lla2ned(40,32,[40,5])
        hd = 8000.0    # destination for flight 1
        dt = 1 # time step in seconds

        # implement the dynamics model
        xdot = v*np.cos(psi)*np.cos(gamma) + self.wind(lat,lon)[0]
        ydot = v*np.sin(psi)*np.cos(gamma) + self.wind(lat,lon)[1]
        hdot = v*np.sin(gamma)
        vdot = (self.Thrust(h)*delta-self.drag(v,h,mu,m))/m - 9.8065*np.sin(gamma)
        psidot = self.lift_coefficient(v,h,mu,m)*124.65*self.rho(h)*v*np.sin(mu)/(2*m)/np.cos(gamma)
        mdot = -self.fuel_consumption(v,delta,h)

        # update states
        x += xdot*dt
        y += ydot*dt
        h += hdot*dt
        v += vdot*dt
        psi += psidot*dt
        m += mdot*dt
        d = np.sqrt((x-xd)**2+(y-yd)**2+(h-hd)**2)
        lat, lon = self.ned2lla(x,y,[40,5])

        # calculate reward, added closing distance reward and normalized the reward
        self.reward = -self.calculate_cost(delta,h,v,d)
        # huge reward for reaching the destination
        if self.reached_destination(d):
            self.reward = 1e5
            terminated = True
            print("Reached destination")

        # impose soft constraints to ensure the agent stays within the bounds
        if lat < 39.0 or lat > 41.0 or lon < 4.9 or lon > 33.0 or h < 7000.0 or h > 13000.0:
            self.reward = -1e5
            terminated = True

        # check if episode is terminated
        self.time += dt
        terminated = self.time > 15000

        self.state = np.array([lat,lon,h,v,psi,m,d],dtype=np.float32)
        
        
        return self.state, self.reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        # Initial conditions for Flight 1 (example)
        # initial distance to goal is 2302443.2
        self.state = np.array([40.0, 5.0, 8000.0, 210.0, 0.0, 68000.0, 2302443.2], dtype=np.float32)
        self.time = 0.0
        self.reward = 0.0
        return self.state, {}

    def render(self):
        pass

    def close(self):
        pass
    
    def calculate_cost(self,delta,h,v,d):
        return d**2/1e14 + 0.05 + self.fuel_consumption(v,delta,h)
    
    def wind(self,lat,lon):
        # Wind coefficients
        # Coefficient of cx
        cx0 = -21.151
        cx1 = 10.0039
        cx2 = 1.1081
        cx3 = -0.5239
        cx4 = -0.1297
        cx5 = -0.006
        cx6 = 0.0073
        cx7 = 0.0066
        cx8 = -0.0001
        # Coefficient of cy
        cy0 = -65.3035
        cy1 = 17.6148
        cy2 = 1.0855
        cy3 = -0.7001
        cy4 = -0.5508
        cy5 = -0.003
        cy6 = 0.0241
        cy7 = 0.0064
        cy8 = -0.000227
        Wx = cx0 + cx1 * lon + cx2 * lat + cx3 * lon * lat + \
         cx4 * lon ** 2 + cx5 * lat ** 2 + cx6 * lon ** 2 * lat + \
         cx7 * lon * lat ** 2 + cx8 * lon ** 2 * lat ** 2
    
        Wy = cy0 + cy1 * lon + cy2 * lat + cy3 * lon * lat + \
            cy4 * lon ** 2 + cy5 * lat ** 2 + cy6 * lon ** 2 * lat + \
            cy7 * lon * lat ** 2 + cy8 * lon ** 2 * lat ** 2
        return Wx,Wy
    
    def drag(self,v,h,mu,m):
        # Drag model
        rho = self.rho(h)
        S = 124.65
        g = 9.8065
        cl = self.lift_coefficient(v,h,mu,m)
        cd = 0.025452 + 0.035815*cl**2
        return (cd*S*rho*v**2)/2
    
    def lift_coefficient(self,v,h,mu,m):
        rho = self.rho(h)
        S = 124.65
        g = 9.8065
        cl = 2*m*g/(rho*v**2*S*np.cos(mu))
        return cl

    def rho(self,h):
        # Air density model
        s = np.sign((1 - (2.2257e-5) * h))
        p = 1.225*s*abs(1 - (2.2257e-5) * h) ** 4.2586
        if np.isnan(p) or p < 0.0:
            p = 0.19
        return p
    
    def fuel_consumption(self,v,delta,h):
        # Fuel consumption constants
        Cf_cr = 0.92958
        Cf_1 = 0.70057
        Cf_2 = 1068.1
        eta = (Cf_1 / 60000) * (1 + 1.943 * v / Cf_2)
        return delta * self.Thrust(h) * eta * Cf_cr
    
    def Thrust(self,h):
        # Thrust model
            # Engine thrust constants
        CT_cr = 0.95
        CTc_1 = 146590
        CTc_2 = 53872
        CTc_3 = 3.0453e-11
        return CT_cr * CTc_1 * (1 - 3.28 * h / CTc_2 + CTc_3 * (3.28 * h) ** 2)

    def reached_destination(self,d):
        return d < 100.0
    
    def lla2ned(self, lat, lon,lla0):
        # convert lat, lon, alt to north, east, down
        earth_radius = 6378137.0
        lat0 = lla0[0]
        lon0 = lla0[1]
        x = (lat-lat0)*earth_radius/180/np.pi
        y = (lon-lon0)*earth_radius*np.cos(lat0*np.pi/180)/180/np.pi
        return x,y
    
    def ned2lla(self, x, y,lla0):
        # convert north, east, down to lat, lon, alt
        earth_radius = 6378137.0
        lat0 = lla0[0]
        lon0 = lla0[1]
        lat = lat0 + x/earth_radius*180/np.pi
        lon = lon0 + y/earth_radius/np.cos(lat0*np.pi/180)*180/np.pi
        return lat,lon
