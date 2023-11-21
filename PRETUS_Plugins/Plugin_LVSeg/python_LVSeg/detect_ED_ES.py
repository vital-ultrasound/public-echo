"""
A script to detect ED and ES in realtime. This is then reimplemented in C++
"""

import numpy as np
import matplotlib.pyplot as plt

datapath = '/home/ag09/data/VITAL/echo/misc'

area = np.load('{}/area_values.npy'.format(datapath))
times = np.load('{}/times_values.npy'.format(datapath))

plt.plot(times, area)
plt.show()

time_buffer, area_buffer = [], []

def process(current_t, current_a):
    time_buffer.append(current_t)
    area_buffer.append(current_a)



# simulate the real time

for i, t in enumerate(times):
    current_t = t
    current_a = area[i]

    process(current_t, current_a)