import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import numpy as np
path = "C:/Users/cihan/Downloads/env_processed.nc"
ds = xr.open_dataset(path)
lats = ds['latitude'].values
lons = ds['longitude'].values
times = np.array(ds['time'].values,dtype='datetime64[s]')
pl = ds['level'].values
pcfa = ds['pcfa'].values
aCCF_merged = ds['aCCF_merged'].values
aCCF_CH4 = ds['aCCF_CH4'].values
aCCF_CO2 = ds['aCCF_CO2'].values
aCCF_Cont = ds['aCCF_Cont'].values
aCCF_dCont = ds['aCCF_dCont'].values
aCCF_H2O = ds['aCCF_H2O'].values
aCCF_merged = ds['aCCF_merged'].values
aCCF_nCont = ds['aCCF_nCont'].values
aCCF_NOx = ds['aCCF_NOx'].values
aCCF_O3 = ds['aCCF_O3'].values

lat = 42.2
lon = 31.1
time = 1
press = 200
interp = RegularGridInterpolator(([0,1,2,3],pl, lats, lons), aCCF_merged)

print(interp([time, press, lat, lon]))