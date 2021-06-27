'''
Refs:
http://databookuw.com/databook.pdf

'''

import numpy as np


def evd_np(X2D, mode_number):

    timestep_number = X2D.shape[0]
    particle_number = X2D.shape[1]

    # Compute mean?
    m1 = np.mean(X2D, axis=0)
    m2 = np.mean(m1, axis=0)
    m = np.ones((timestep_number, particle_number)) * m2

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

    # # Compute subspace 2D mapped data
    # Y2D = np.matmul(X2D, W)

    # Instead of returning mapped data, return actual data while reduced
    X2D = X2D[:, idx]
    Y2D = X2D[:, :mode_number]

    return idx, W, Y2D