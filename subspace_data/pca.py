'''
Refs:
http://databookuw.com/databook.pdf

'''

import numpy as np
import matplotlib.pyplot as plt

LOCATION = './learning_to_simulate/datasets/Excavation_PCA/'

def evd_np(X2D):

    observations = X2D.shape[0]
    dimensions = X2D.shape[1]

    # Compute mean (?)
    m1 = np.mean(X2D, axis=0)
    m2 = np.mean(m1, axis=0)
    m = np.ones((observations, dimensions)) * m2

    # Get mean centered data
    Xn = np.subtract(X2D, m)

    # Compute covariance matrix
    C = np.matmul(np.transpose(Xn), Xn) / (dimensions-1)

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(C)

    # Sort eigenvalues and eigenvectors
    idx = evals.argsort()[::-1]  
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute transformation matrix
    # W = evecs[:, :mode_number]

    energy = []
    for i in range(1, dimensions):
        energy.append(np.sum(evals[:i]) / np.sum(evals))
    plt.plot(range(1, dimensions), energy, '.-')
    plt.savefig(LOCATION + 'energy.png', dpi=500)

    return evecs, evals, m2