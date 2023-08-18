import numpy as np
import matplotlib.pyplot as plt
from dynamicaleqns import dynamics
from dynamicaleqns import wind


psi = np.zeros(1001)
h = np.zeros(1001)
v= np.zeros(1001)
m = np.zeros(1001)
gama = 0.1
psi[0] = 0
m[0] = 68000
del_t = 1
x = np.zeros(1001)
y = np.zeros(1001)
x[0] = 0
y[0] = 0
h[0] = 8000
v[0] = 210
mu = 0
d = 0.7
t = np.arange(1, 1001)
for i in range(1000):
    result = dynamics(x[i], y[i], h[i], v[i], psi[i], m[i], gama, mu, 1, del_t)
    x[i+1], y[i+1], h[i+1], v[i+1], psi[i+1], m[i+1] = result
    if m[i] < 20000:
        break
# # Create a figure and multiple subplots
# fig, axs = plt.subplots(6, 1, figsize=(8, 12))

# # Plot y1 in the first subplot
# axs[0].plot(x, y, label='x-y')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# axs[0].legend()

# # Plot y2 in the second subplot
# axs[1].plot(t, h, label='h-time')
# axs[1].set_xlabel('time')
# axs[1].set_ylabel('h')
# axs[1].legend()

# # Plot y3 in the third subplot
# axs[2].plot(t, v, label='v-time')
# axs[2].set_xlabel('time')
# axs[2].set_ylabel('v')
# axs[2].legend()

# axs[3].plot(t, m, label='mass-time')
# axs[3].set_xlabel('time')
# axs[3].set_ylabel('m')
# axs[3].legend()



# # Adjust layout to prevent overlapping of subplots
# plt.tight_layout()

# # Show the plots
# plt.show()
print(m)