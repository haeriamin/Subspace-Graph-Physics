'''
Refs:
https://en.wikipedia.org/wiki/Principal_component_analysis
https://towardsdatascience.com/svd-in-machine-learning-pca-f25cf9b837ae

'''

import numpy as np


def evd_np(X2D, mode_number):

    particle_number = X2D.shape[1]

    # Compute mean
    m = np.mean(X2D, axis=0)

    # Get mean centered data
    Xn = np.subtract(X2D, m)

    # Compute covariance matrix
    C = np.matmul(np.transpose(Xn), Xn) / (particle_number-1)

    # Compute eigenvalues anc eigenvectors
    evals, evecs = np.linalg.eig(C)

    # Sort eigenvalues and eigenvectors
    idx = evals.argsort()[::-1]  
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute transformation matrix
    W = evecs[:, :mode_number]

    # Compute subspace 2D data
    Y2D = np.matmul(X2D, W)

    return idx, W, Y2D