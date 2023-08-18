import numpy as np
import matplotlib.pyplot as plt

def dynamics(x,y,h,v,psi,m,gama,mu,d,del_t):

    if v < 50:
        v = 50

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

    # dynamics

    xdot = v*np.cos(psi)*np.cos(gama) + wind(x,y)[0]
    ydot = v*np.sin(psi)*np.cos(gama) + wind(x,y)[1]
    hdot = v*np.sin(gama)
    vdot = thrmax*d/m - 0.5*rho*v**2*S*cd/m - g*np.sin(gama)
    psidot = cl*S*rho*v*np.sin(mu)/(2*m)/np.cos(gama)
    mdot = -f

    x = x + xdot*del_t
    y = y + ydot*del_t
    h = h + hdot*del_t
    v = v + vdot*del_t
    psi = psi + psidot*del_t
    m = m + mdot*del_t

    return x,y,h,v,psi,m









def wind(x,y):
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
    return np.array([wx,wy])