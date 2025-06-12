import os
import numpy as np
from physics_simulator.pendulum import *

# Pure signal
pendulum = DampedPendulum()
t, y = pendulum.solve()

# Gaussian Noise
mean = 0.0
sigma = 0.1
noise = np.random.normal(loc=mean, scale=sigma, size=(len(y), len(t)))

# Signal assembling
noisy_signal = np.array(y) + noise

# Concatenate time and signal
noisy_experiment = np.vstack((t, noisy_signal))

# Saving
np.save("noisy_xp.npy", noisy_experiment)
