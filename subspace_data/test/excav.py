'''
Generate excavation data

Example usage (from parent directory):
'python -m subspace_data.test.excav_sim'

'''

import numpy as np


def sim():
    # Number of observations should be more than variables (dimensions)
    time = 2       # [sec]
    rate = 60      # [Hz]
    p = 16         # number of particles
    dt = 1/rate

    n = int(time*rate)  # number of observations
    m = 2*p        # number of dimensions

    # State vector @ t=0 in [m]
    X = np.empty(shape=(n, p, 2), dtype=np.float)

    X[0, 0, :] = [0, 0]
    X[0, 1, :] = [1, 0]
    X[0, 2, :] = [2, 0]
    X[0, 3, :] = [3, 0]

    X[0, 4, :] = [0, 1]
    X[0, 5, :] = [1, 1]
    X[0, 6, :] = [2, 1]
    X[0, 7, :] = [3, 1]

    X[0, 8, :] = [0, 2]
    X[0, 9, :] = [1, 2]
    X[0, 10, :] = [2, 2]
    X[0, 11, :] = [3, 2]

    X[0, 12, :] = [0, 3]
    X[0, 13, :] = [1, 3]
    X[0, 14, :] = [2, 3]
    X[0, 15, :] = [3, 3]

    # Simulate excavation: constant accelerated motion x = x0 + v(t) * t
    scale = 1/100
    power = 1/100
    a = -1
    b = 1

    for i in range(1, n):
        # noise = ((b-a)*np.random.rand() + a) * power
        noise = 0
        
        X[i, 0, :] = X[i-1, 0, :] + np.multiply(scale*(i/rate), [.9, -.7]) + noise
        X[i, 1, :] = X[i-1, 1, :] + np.multiply(scale*(i/rate), [.6, -.4]) + noise
        X[i, 2, :] = X[i-1, 2, :] + np.multiply(scale*(i/rate), [.3, -.1]) + noise
        X[i, 3, :] = X[i-1, 3, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise

        X[i, 4, :] = X[i-1, 4, :] + np.multiply(scale*(i/rate), [1., 0.]) + noise
        X[i, 5, :] = X[i-1, 5, :] + np.multiply(scale*(i/rate), [.8, 0.]) + noise
        X[i, 6, :] = X[i-1, 6, :] + np.multiply(scale*(i/rate), [.6, 0.]) + noise
        X[i, 7, :] = X[i-1, 7, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise

        X[i, 8, :] = X[i-1, 8, :] + np.multiply(scale*(i/rate), [.9, .5]) + noise
        X[i, 9, :] = X[i-1, 9, :] + np.multiply(scale*(i/rate), [.6, .2]) + noise
        X[i, 10, :] = X[i-1, 10, :] + np.multiply(scale*(i/rate), [.3, 0.]) + noise
        X[i, 11, :] = X[i-1, 11, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise
        
        X[i, 12, :] = X[i-1, 12, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise
        X[i, 13, :] = X[i-1, 13, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise
        X[i, 14, :] = X[i-1, 14, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise
        X[i, 15, :] = X[i-1, 15, :] + np.multiply(scale*(i/rate), [0., 0.]) + noise

    return X
