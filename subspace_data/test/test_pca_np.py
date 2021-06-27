'''
Example usage (from parent directory):
'python -m subspace_data.test.test_pca_np'

Refs:
http://databookuw.com/databook.pdf
https://en.wikipedia.org/wiki/Principal_component_analysis
https://towardsdatascience.com/svd-in-machine-learning-pca-f25cf9b837ae
https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from subspace_data.test import excav


def plot(X, Y, no):
    f = plt.figure(1)

    def get_cmap(n, name='jet'):
        return plt.cm.get_cmap(name, n)

    # Option 0:
    ax1 = plt.subplot(1, 2, 1)
    cmap = get_cmap(len(X[0]))
    for i in range(len(X)):
        for j in range(len(X[0])):
            ax1.plot(X[i, j, 0], X[i, j, 1], '.', c=cmap(j))
    ax1.set_xlim([-0.2, 3.2])
    ax1.set_ylim([-1.2, 3.2])
    ax1.axis('equal')

    ax2 = plt.subplot(1, 2, 2)
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            ax2.plot(Y[i, j, 0], Y[i, j, 1], '.', c=cmap(j))
    ax2.set_xlim([-0.2, 3.2])
    ax2.set_ylim([-1.2, 3.2])
    ax2.axis('equal')

    # Option 1:
    # cmap = get_cmap(len(X[0]))
    # I = range(no)
    # plt.plot(I, X[:, 0], '.', c=cmap(0))
    # plt.plot(I, X[:, 1], '.', c=cmap(1))
    # plt.plot(I, X[:, 2], '.', c=cmap(2))
    # plt.plot(I, X[:, 3], '.', c=cmap(3))
    # plt.plot(I, X[:, 4], '.', c=cmap(4))
    # plt.plot(I, X[:, 5], '.', c=cmap(5))
    # plt.plot(I, X[:, 6], '.', c=cmap(6))
    # plt.plot(I, X[:, 7], '.', c=cmap(7))
    # plt.plot(I, X[:, 8], '.', c=cmap(8))
    # plt.plot(I, X[:, 9], '.', c=cmap(9))
    # plt.plot(I, X[:, 10], '.', c=cmap(10))
    # plt.plot(I, X[:, 11], '.', c=cmap(11))
    # plt.plot(I, X[:, 12], '.', c=cmap(12))
    # plt.plot(I, X[:, 13], '.', c=cmap(13))
    # plt.plot(I, X[:, 14], '.', c=cmap(14))
    # plt.plot(I, X[:, 15], '.', c=cmap(15))
    # plt.plot(I, Y[:, 0], 'x', c=cmap(0))
    # plt.plot(I, Y[:, 1], 'x', c=cmap(1))
    # plt.plot(I, Y[:, 2], 'x', c=cmap(2))
    # plt.plot(I, Y[:, 3], 'x', c=cmap(3))
    # plt.plot(I, Y[:, 4], 'x', c=cmap(4))
    # plt.plot(I, Y[:, 5], 'x', c=cmap(5))
    # plt.plot(I, Y[:, 6], 'x', c=cmap(6))
    # plt.plot(I, Y[:, 7], 'x', c=cmap(7))
    # plt.xlabel("Timestep")
    # plt.ylabel("Position")
    # plt.legend(['Full1', 'Full2', 'Full3', 'Sub1', 'Sub2', 'Sub3'])

    plt.savefig('test_pca_evd_np.png', dpi=1000)
    f.show()


def pca_svd(X2D, mode_number, X2D_main):

    timestep_number = X2D.shape[0]
    particle_number = X2D.shape[1]

    # Compute mean
    m = np.mean(X2D, axis=0)

    # Get mean subtracted data
    Xn = np.subtract(X2D, m)

    U, s, VT = np.linalg.svd(Xn)
    W = VT.T[:, :mode_number]
    Y2D = np.matmul(X2D_main, W)

    energy = []
    for i in range(1, particle_number):
        energy.append(np.sum(s[:i])/np.sum(s))

    f = plt.figure(2)
    plt.plot(range(1, particle_number), energy, '.-')
    f.show()

    return 1, W, Y2D


def pca_evd(X2D, mode_number, X2D_main):
    eps = 1e-13

    timestep_number = X2D.shape[0]
    particle_number = X2D.shape[1]

    # Compute mean
    m = np.mean(X2D, axis=0)

    # Get mean subtracted data
    Xn = np.subtract(X2D, m)

    # f = plt.figure(4)
    # I = range(timestep_number)
    # plt.plot(I, Xn[:, 0], '.-')
    # plt.plot(I, Xn[:, 1], '.-')
    # plt.plot(I, Xn[:, 7], '.-')
    # plt.plot(I, Xn[:, 9], '.-')
    # plt.legend(['0', '1', '7', '9'])
    # f.show()

    # Compute covariance matrix
    C = np.matmul(np.transpose(Xn), Xn) / timestep_number

    # To avoid negative eigenvalues
    for i in range(particle_number):
        C[i][i] += eps

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(C)

    # Modify eigenvalues
    for i in range(len(evals)):
        if evals.real[i] == eps:
            evals[i] = 0.0
    # evals = evals.real
    print(evals)

    # Sort eigenvalues and eigenvectors
    idx = evals.argsort()[::-1]  
    evals = evals[idx]
    evecs = evecs[:, idx]
    print(idx)

    # Compute transformation matrix
    W = evecs[:, :mode_number]

    # Compute subspace 2D mapped data
    # Y2D = np.matmul(X2D_main, W)

    # Instead of returning mapped data, return actual data while reduced
    X2D_main = X2D_main[:, idx]
    Y2D = X2D_main[:, :mode_number]

    energy = []
    for i in range(1, particle_number):
        energy.append(np.sum(evals[:i]) / np.sum(evals))

    f = plt.figure(2)
    plt.plot(range(1, particle_number), energy, '.-')
    f.show()

    f = plt.figure(3)
    ax = sns.heatmap(C, annot=True, cmap='Spectral')
    f.show()

    return idx, W, Y2D


if __name__ == '__main__':

    MODE_NUMBER = 8

    # Generate dummy 2D data:
    # step_num = 100
    # data1 = np.random.uniform(low=0., high=100., size=step_num)
    # data2 = np.multiply(5., data1) + np.random.uniform(low=0., high=100., size=step_num)
    # data3 = np.multiply(10., data1) - np.random.uniform(low=0., high=100., size=step_num)
    # X2D = np.stack([data1, data2, data3])
    # X2D = np.transpose(X2D)

    # Generate 2D dummy 3D data:
    # X3D = [[[1, 2], [10, 20], [100, 200]],
    #        [[3, 4], [30, 40], [300, 400]],
    #        [[5, 6], [50, 60], [500, 600]]]
    # print(X3D)

    # Generate 2D excavation 3D data:
    X3D = excav.sim()
    # print(X3D)

    timestep_number = len(X3D)
    print(timestep_number)
    particle_number = len(X3D[0])
    print(particle_number)
    dim_number = len(X3D[0][0])
    print(dim_number)

    # Convert 3D data to 2D data
    X2D_main = np.zeros((dim_number*timestep_number, particle_number))
    X2D = np.zeros((dim_number*timestep_number, particle_number))
    for i in range(timestep_number):
        for j in range(particle_number):
            for k in range(dim_number):
                X2D_main[dim_number*i+k][j] = X3D[i][j][k]
                X2D[dim_number*i+k][j] = np.abs(X3D[i][j][k]-X3D[0][j][k])
                # if i == 0:
                #     X2D[dim_number*i+k][j] = 0.
                # else:
                #     X2D[dim_number*i+k][j] = np.abs(X3D[i][j][k]-X3D[i-1][j][k])*60

    idx, W, Y2D = pca_evd(X2D, MODE_NUMBER, X2D_main)
    # idx, W, Y2D = pca_svd(X2D, MODE_NUMBER, X2D_main)

    # Convert 2D data to 3D data
    Y3D = np.zeros((timestep_number, MODE_NUMBER, dim_number))
    for i in range(dim_number*timestep_number):
        for j in range(MODE_NUMBER):
            Y3D[i // 2][j][i % 2] = Y2D[i][j]
    # print(Y3D)

    # Convert 3D data to 1D data
    l = 0
    Y1D = np.zeros((timestep_number*MODE_NUMBER*dim_number))
    for i in range(timestep_number):
        for j in range(MODE_NUMBER):
            for k in range(dim_number):
                Y1D[l] = Y3D[i][j][k]
                l += 1
    # print(Y1D)

    # Convert 3D data to 2D data
    Y2D_2 = np.zeros((timestep_number, MODE_NUMBER*dim_number))
    for i in range(timestep_number):
        for j in range(MODE_NUMBER):
            for k in range(dim_number):
                Y2D_2[i][dim_number*j+k] = Y3D[i][j][k]
    # print(Y2D_2)

    # plot(X2D_main, Y2D, 2*timestep_number)
    plot(X3D, Y3D, timestep_number)
    input()

    ## Test1:
    # G = [1, 2, 10, 20, 100, 200, 3, 4, 30, 40, 300, 400, 5, 6, 50, 60, 500, 600]
    # position_shape = [3, -1, 2]
    # parsed_features = np.reshape(G, position_shape)
    # print(parsed_features)

    ## Test2:
    # import struct
    # print(struct.pack('q', 6))
    # print(struct.pack('d', 612.1312211))
    # a = [612.1312211, 612.1312211]
    # bufa = struct.pack('%sd' % len(a), *a)
    # print(bufa)
    # b = [6, 6]
    # bufb = struct.pack('%sq' % len(b), *b)
    # print(bufb)
