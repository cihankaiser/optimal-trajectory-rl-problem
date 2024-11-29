import gymnasium as gym
import numpy as np
from gymnasium import spaces
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

class AircraftEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32),
                                       high=np.array([1, 1, 1], dtype=np.float32),
                                       dtype=np.float32)
        self.min_actions = np.array([-0.0261799, -0.5259, 0.5], dtype=np.float32)
        self.max_actions = np.array([0.0261799, 0.5259, 1], dtype=np.float32)
        # states are: latitude, longitude, altitude, speed, heading, mass, distance, relative bearing, remaining fuel, time to destination
        self.min_states = np.array([-10000.0, -10000.0, 7000.0, 150.0, -2*np.pi, 10000.0, 0.0,-np.pi, 0.0, 0.0], dtype=np.float32)
        self.max_states = np.array([10000.0, 2400000.0, 14000.0, 300.0, 2*np.pi, 69000.0, 2400.000, np.pi, 1.0, 12000.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.state = None
        self.time = None

        self.stage = 0  # Current curriculum stage
        self.max_stage = 3  # Maximum number of stages
        self.success_count = 0  # Counter for successful episodes
        self.success_threshold = 25 # Number of successful episodes to progress to the next stage
        path = "C:/Users/cihan/Downloads/env_processed.nc"
        ds = xr.open_dataset(path)
        self.lats = ds['latitude'].values
        self.lons = ds['longitude'].values
        self.times = ds['time'].values
        self.pl = ds['level'].values
        self.pcfa = ds['pcfa'].values
        self.aCCF_merged = ds['aCCF_merged'].values
        self.aCCF_CH4 = ds['aCCF_CH4'].values
        self.aCCF_CO2 = ds['aCCF_CO2'].values
        self.aCCF_Cont = ds['aCCF_Cont'].values
        self.aCCF_dCont = ds['aCCF_dCont'].values
        self.aCCF_H2O = ds['aCCF_H2O'].values
        self.aCCF_merged = ds['aCCF_merged'].values
        self.aCCF_nCont = ds['aCCF_nCont'].values
        self.aCCF_NOx = ds['aCCF_NOx'].values
        self.aCCF_O3 = ds['aCCF_O3'].values
        self.atr_total = np.zeros((4,8,21,71))

    def normalize_state(self, state):
        return 2 * (state - self.min_states) / (self.max_states - self.min_states) - 1

    def denormalize_state(self, normalized_state):
        return (normalized_state + 1) * (self.max_states - self.min_states) / 2 + self.min_states

    def step(self, action):
        delta = 0.5 * (action[2] + 1) * (self.max_actions[2] - self.min_actions[2]) + self.min_actions[2]
        gamma = 0.5 * (action[0] + 1) * (self.max_actions[0] - self.min_actions[0]) + self.min_actions[0]
        mu = 0.5 * (action[1] + 1) * (self.max_actions[1] - self.min_actions[1]) + self.min_actions[1]

        x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time = self.denormalize_state(self.state)
        dtemp = d
        mtemp = m
        m0 = 68000.0  # initial mass
        mf = 32000.0  # fuel mass
        lat, lon = self.ned2lla(x, y, [40, 5])
        xd, yd = self.lla2ned(40, 32, [40, 5])
        hd = 8000.0  # destination for flight 1
        dt = 1  # time step in seconds
        dtt = 1  # time step for integration
        dyn_param = [x, y, h, v, psi, m]

        # rk4 integration
        k1 = self.dynamic_model(dyn_param, action)
        k2 = self.dynamic_model(dyn_param + dtt / 2 * k1, action)
        k3 = self.dynamic_model(dyn_param + dtt / 2 * k2, action)
        k4 = self.dynamic_model(dyn_param + dtt * k3, action)
        dyn_param += dtt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x,y,h,v,psi,m = dyn_param

        d = np.sqrt((x - xd) ** 2 + (y - yd) ** 2 + (h - hd) ** 2)/1e3
        lat, lon = self.ned2lla(x, y, [40, 5])
        rel_psi = self.compute_relative_bearing(lat, lon, 40, 32, psi)
        rem_fuel = 1 - (m0 - m)/mf
        dest_time = 1e3*d / v

        # clip the states to the boundaries
        x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time = np.clip([x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time], self.min_states, self.max_states)

        # calculate climate cost
        dist = abs(d-dtemp)
        fuel = mtemp - m
        atr,idx = self.calculate_atr(dist, fuel, h, v , lat, lon, self.time)
        self.atr_total[idx] += atr

        terminated = False
        truncated = False
        
        if self.reached_destination(d,h):
            reward = 1e5     # Increased the reward for reaching the destination
            terminated = True
            print("Reached destination, lat:",lat,"lon:",lon,"h:",h, "time:",self.time, "total atr:",np.sum(self.atr_total))

        elif np.isnan(lat) or np.isnan(lon) or np.isnan(h) or np.isnan(v) or np.isnan(psi) or np.isnan(m) or np.isnan(d) or np.isnan(rel_psi) or np.isnan(rem_fuel) or np.isnan(dest_time):
            terminated = True
            reward = -1e3  
            print("Terminated due to NaN values")

        elif np.any(np.array([x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time]) <= self.min_states) or np.any(np.array([x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time]) >= self.max_states):
            reward = -1e2  # Penalty for violating boundaries
        else:
            reward = -self.calculate_cost(delta,lat, lon, h, v, d,atr)*0.1


        if terminated and reward > 0:
            self.success_count += 1
            if self.success_count >= self.success_threshold and self.stage < self.max_stage:
                self.stage += 1
                self.success_count = 0  # Reset the counter
                print(f"Stage {self.stage} reached")

        self.time += dt
        if self.time > 15000:
            truncated = True
        self.state = self.normalize_state(np.array([x, y, h, v, psi, m, d, rel_psi, rem_fuel, dest_time], dtype=np.float32))

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        rel_psi = self.compute_relative_bearing(40, 5, 40, 32, 0)
        dest_time = 2299866.9/210
        self.state = self.normalize_state(np.array([0.0, 0.0, 8000.0, 210.0, 0.0, 68000.0, 2299.8669, rel_psi, 1, dest_time], dtype=np.float32))
        self.time = 0.0
        self.atr_total = np.zeros((4,8,21,71))
        return self.state, {}

    def render(self):
        pass

    def close(self):
        pass

    def dynamic_model(self, dyn_param, action):
        delta = 0.5 * (action[2] + 1) * (self.max_actions[2] - self.min_actions[2]) + self.min_actions[2]
        gamma = 0.5 * (action[0] + 1) * (self.max_actions[0] - self.min_actions[0]) + self.min_actions[0]
        mu = 0.5 * (action[1] + 1) * (self.max_actions[1] - self.min_actions[1]) + self.min_actions[1]

        x, y, h, v, psi, m = dyn_param
        lat, lon = self.ned2lla(x, y, [40, 5])
        wx, wy = self.wind(lat, lon)

        xdot = v * np.cos(psi) * np.cos(gamma) + wx
        ydot = v * np.sin(psi) * np.cos(gamma) + wy
        hdot = v * np.sin(gamma)
        vdot = (self.Thrust(h) * delta - self.drag(v, h, mu, m)) / m - 9.8065 * np.sin(gamma)
        psidot = self.lift_coefficient(v, h, mu, m) * 124.65 * self.rho(h) * v * np.sin(mu) / (2 * m) / np.cos(gamma)
        mdot = -self.fuel_consumption(v, delta, h)

        return np.array([xdot, ydot, hdot, vdot, psidot, mdot], dtype=np.float32)
    
    
    def calculate_cost(self, delta, lat, lon, h, v, d, atr):

        # Distance penalty
        distance_penalty = (d)**2
        
        # Fuel consumption penalty
        fuel_penalty = self.fuel_consumption(v, delta, h)
        
        # Velocity penalty (if deviating from a target speed, e.g., 250 m/s)
        target_velocity = 240.0
        velocity_penalty = np.sqrt((v - target_velocity)**2) / target_velocity
        lat_pen = np.sqrt((lat-40)**2)
        lon_pen = np.sqrt((lon-32)**2)
        altitude_penalty = np.sqrt((h-8000)**2)

        bearing_penalty = np.sqrt((self.compute_relative_bearing(lat, lon, 40, 32, 0))**2)
        # Combined cost
        cost =  fuel_penalty + distance_penalty*1e-6 + bearing_penalty*1e-3 + lat_pen*1e-3 + lon_pen*1e-3 +altitude_penalty*1e-4 + velocity_penalty + 2 #atr*1e10 # + dest_time*1e-3 
        
        return cost
    
    def wind(self, lat, lon):
        cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8 = -21.151, 10.0039, 1.1081, -0.5239, -0.1297, -0.006, 0.0073, 0.0066, -0.0001
        cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8 = -65.3035, 17.6148, 1.0855, -0.7001, -0.5508, -0.003, 0.0241, 0.0064, -0.000227
        
        Wx = cx0 + cx1 * lon + cx2 * lat + cx3 * lon * lat + \
             cx4 * lon ** 2 + cx5 * lat ** 2 + cx6 * lon ** 2 * lat + \
             cx7 * lon * lat ** 2 + cx8 * lon ** 2 * lat ** 2
        Wy = cy0 + cy1 * lon + cy2 * lat + cy3 * lon * lat + \
             cy4 * lon ** 2 + cy5 * lat ** 2 + cy6 * lon ** 2 * lat + \
             cy7 * lon * lat ** 2 + cy8 * lon ** 2 * lat ** 2
        
        return Wx, Wy

    def Thrust(self, h):
        #engine thrust constants
        CT_cr = 0.95
        CTc_1 = 146590
        CTc_2=53872
        CTc_3 = 3.0453e-11 
        Thrust_max = CT_cr*CTc_1*(1 - 3.28*h/CTc_2 + CTc_3*(3.28*h)**2)
        return Thrust_max
    
    def fuel_consumption(self, v, delta, h):
        #fuel consumption constants
        Cf_cr=0.92958
        Cf_1=0.70057
        Cf_2=1068.1
        eta = (Cf_1/60000)*(1 + 1.943*v/Cf_2)
        Thr_max = self.Thrust(h)
        f = delta*Thr_max*eta*Cf_cr
        return f
    
    def rho(self, h):
        rho0 = 1.225
        rho = rho0 * abs(1 - (2.25577e-5) * h) ** 4.2586
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
        """
        Convert latitude and longitude to North-East-Down (NED) coordinates relative to a reference point.
        
        Parameters:
        - lat: Latitude in degrees
        - lon: Longitude in degrees
        - ref: Reference point (lat0, lon0) in degrees
        
        Returns:
        - x: North coordinate in meters
        - y: East coordinate in meters
        """
        lat0, lon0 = np.radians(ref)  # Convert reference point to radians
        lat, lon = np.radians([lat, lon])  # Convert latitude and longitude to radians

        Re = 6371000  # Mean radius of the Earth in meters

        x = Re * (lat - lat0)  # North coordinate
        y = Re * np.cos(lat0) * (lon - lon0)  # East coordinate

        return x, y

    def ned2lla(self, x, y, ref):
        """
        Convert North-East-Down (NED) coordinates to latitude and longitude relative to a reference point.
        
        Parameters:
        - x: North coordinate in meters
        - y: East coordinate in meters
        - ref: Reference point (lat0, lon0) in degrees
        
        Returns:
        - lat: Latitude in degrees
        - lon: Longitude in degrees
        """
        lat0, lon0 = np.radians(ref)  # Convert reference point to radians

        Re = 6371000  # Mean radius of the Earth in meters

        lat = lat0 + x / Re  # Latitude in radians
        lon = lon0 + y / (Re * np.cos(lat0))  # Longitude in radians

        lat, lon = np.degrees([lat, lon])  # Convert back to degrees

        return lat, lon
    
    def reached_destination(self, d,h):
        if self.stage == 0:
            c = 50
        elif self.stage == 1:
            c = 25
        elif self.stage == 2:
            c = 10
        elif self.stage == 3:
            c = 5
        else:
            c = 1
        return bool(d < 50 )#and h <8100) # and lat < 40.1 and lat > 39.9 and lon < 32.1 and lon > 31.9

    def compute_relative_bearing(self, lat, lon, dest_lat, dest_lon, psi):
        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat, lon, dest_lat, dest_lon])
        
        # Compute the difference in longitudes
        d_lon = lon2 - lon1
        
        # Compute the desired heading
        y = np.sin(d_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        psi_dest = np.arctan2(y, x)
        
        # Compute the relative bearing
        relative_bearing = psi_dest - psi
        relative_bearing = np.arctan2(np.sin(relative_bearing), np.cos(relative_bearing))  # Normalize to [-pi, pi]
        
        return relative_bearing
    
    def reflect_boundary(self, value, min_val, max_val):
        if value < min_val:
            return min_val + (min_val - value)
        elif value > max_val:
            return max_val - (value - max_val)
        return value
    
    def calculate_atr(self,  dist, fuel, h, v , lat, lon,time):

        nox = self.calculate_nox_emissions(h, v, fuel)
        lat_idx, lon_idx = self.find_lla_indices(self.lats, self.lons, lat, lon)
        time_idx = self.determine_time_index(time)
        press = self.calculate_pressure_levels(h)
        pl_idx = self.determine_pressure_level(press)

        # Calculate separate ATRs and ATR from merged a_CCF
        atr_nox = self.aCCF_NOx[time_idx, pl_idx, lat_idx, lon_idx] * nox
        atr_h2o = self.aCCF_H2O[time_idx, pl_idx, lat_idx, lon_idx] * fuel
        atr_co2 = self.aCCF_CO2[time_idx, pl_idx, lat_idx, lon_idx] * fuel
        atr_con = self.aCCF_dCont[time_idx, pl_idx, lat_idx, lon_idx] * self.pcfa[time_idx, pl_idx, lat_idx, lon_idx] * 1e-3 * dist  

        #interp = RegularGridInterpolator(([0,1,2,3],pl, lats, lons), aCCF_merged)

        idx = (time_idx, pl_idx, lat_idx, lon_idx)
        # Calculate total ATR
        atr_total = atr_co2 + atr_h2o + atr_nox + atr_con

        return atr_total, idx
    
    def calculate_nox_emissions(self, h, v, fuel, deltaT=0):
        # Standart atmospheric temperature at MSL
        T_0 = 288.15
        
        # Adiabatic index of air
        k = 1.4
        
        # Real gas constant for air
        r = 287.05287
        
        # Gravitational acceleration
        g = 9.80665
        
        # ISA temperature gradient with altitude below the tropopause
        b = -0.0065
        
        # Standart atmospheric pressure at MSL
        p_0 = 101325
        
        # Determination of temperature for non-ISA conditions
        T = T_0 + deltaT + b * h
        
        # Determination of pressure for non-ISA conditions
        p = p_0 * ((T - deltaT) / T_0) ** (-g / (b * r))
        
        # Determination of pressure above tropopause for non-ISA conditions
    #     if h > 11000:
    #         p = p * exp(-g * (h - 11000) / (r * (T - deltaT)))
        
        # Determination of density for non-ISA conditions
        rho = p / (r * T)
        
        # Determination of speed of sound for non-ISA conditions
        a = np.sqrt(k * r * T)
        
        # Determination of mach numbers
        machs = v / a
        
        # Calculate total pressure and temperature for the corresponding actual values
        t_total = T * (1 + 0.2 * machs ** 2)
        p_total = p * (1 + 0.2 * machs ** 2) ** 3.5
        
        # Calculate correction values delta and theta from total values
        delta = p_total / p_0
        theta = t_total / T_0
        
        # Calculate fuel consumed in each time interval
        f_a = fuel / 1.65625  # delta_fuel/delta_t
        
        # Calculate actual fuel flow rate at altitude
        f_ref = f_a / (delta * np.sqrt(theta))
        
        # Interpolate (least squares, second order) CFM567B24E fref-EINOx ICAO data and calculate real emission indexes of NOx by our given fref
        einox_icao = np.array([4.4, 10.1, 20.5, 25.3])  # idle, approach, climb out, take off conditions respectively
        fref_icao = np.array([0.103, 0.308, 0.895, 1.086])
        
        # Interpolating
        ip_coeff = np.linalg.lstsq(np.vstack([fref_icao, np.ones(len(fref_icao))]).T, einox_icao, rcond=None)[0]
        
        einox_ref = ip_coeff[0] * f_ref + ip_coeff[1]
        
        q = 1e-3 * np.exp(-0.0001426 * (h - 12900))  # Specific humidity at the given altitude
        h_c = np.exp(-19.0 * (q - 0.00634))  # Humidity correction factor
        
        # Calculate actual emission index factor of NOx
        einox_actual = einox_ref * (delta ** 0.4) * (theta ** 3) * h_c
        
        # Calculate NOx emissions in kgs
        nox = fuel * einox_actual * 1e-3
        
        return nox
    
    def calculate_pressure_levels(self,h):
        """
        Calculate pressure levels from altitudes.

        Args:
            h (ndarray): Array of altitudes (in meters).

        Returns:
            ndarray: Array of pressure levels (in hPa).
        """
        return (100 * ((44331.514 - h) / 11880.516) ** (1 / 0.1902632)) / 100
    
    def find_lla_indices(self,lats,lons,lat,lon):

        idx_closest_lat = np.argmin(abs(lats - lat))
        idx_closest_lon = np.argmin(abs(lons - lon))

        # Find the indices where the minimum value occurs    
        return [idx_closest_lat, idx_closest_lon]
    
    def determine_time_index(self,time):
        """
        Determine the time index based on the time of the day.

        Args:
            time (float): Time in seconds.

        Returns:
            int: Time index.
        """
        if time < 3600:    # 9:00 - 10:00
            return 0
        elif 3600 <= time < 7200:   # 10:00 - 11:00
            return 1
        else:                        # Last part of the flight
            return 2

    def determine_pressure_level(self,press):
        """
        Determine the pressure level index based on the pressure.

        Args:
            press (float): Pressure in hPa.

        Returns:
            int: Pressure level index.
        """
        if press < 150:
            return 0
        elif 150 <= press < 175:
            return 1
        elif 175 <= press < 200:
            return 2
        elif 200 <= press < 225:
            return 3
        elif 250 <= press < 300:
            return 4
        elif 300 <= press < 350:
            return 5
        elif 350 <= press < 400:
            return 6
        else:
            return 7
        

    # Register the environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id='Aircraft-v0',
    entry_point='environment_norm:AircraftEnv',
)