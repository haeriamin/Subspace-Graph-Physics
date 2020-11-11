'''
Example usage (from parent directory):
'python -m subspace_data.test.test_pca_evd_np'

Refs:
https://en.wikipedia.org/wiki/Principal_component_analysis
https://towardsdatascience.com/svd-in-machine-learning-pca-f25cf9b837ae

'''

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np


# x = tf.random.uniform([3, 3])
# print("Is there a GPU available? "),
# print(tf.config.experimental.list_physical_devices("GPU"))
# print("Is the Tensor on GPU #0?  "),
# print(x.device.endswith('GPU:0'))


def pca_evd_np(X2D, mode_number):

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


def plot(X, Y, no):

    # Option 1:
    I = range(no)
    plt.plot(I, X[:, 0], '-.', color='b')
    plt.plot(I, X[:, 1], '-.', color='r')
    plt.plot(I, X[:, 2], '-.', color='k')
    plt.plot(I, Y[:, 0], '-+', color='k')
    # plt.plot(I, Y[:, 1], '-x', color='r')
    # plt.plot(I, Y[:, 2], '-|', color='b')
    plt.xlabel("Timestep")
    plt.ylabel("Position")
    plt.legend(['Full1', 'Full2', 'Full3', 'Sub1', 'Sub2', 'Sub3'])

    # Option 2:
    # def axisEqual3D(ax):
    #     extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    #     sz = extents[:,1] - extents[:,0]
    #     centers = np.mean(extents, axis=1)
    #     maxsize = max(abs(sz))
    #     r = maxsize/2
    #     for ctr, dim in zip(centers, 'xyz'):
    #         getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Xn_val[:,0],Xn_val[:,1],Xn_val[:,2], marker='.')
    # ax.scatter(Yn_val[:,2],Yn_val[:,1],Yn_val[:,0], marker='+')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.ylabel("Z")
    # axisEqual3D(ax)
    # plt.legend(['Full', 'Sub'])

    plt.savefig('test_pca_evd_np.png', dpi=1000)
    plt.show()


if __name__ == '__main__':

    MODE_NUMBER = 1

    # Generate dummy 2D data:
    # step_num = 100
    # data1 = np.random.uniform(low=0., high=100., size=step_num)
    # data2 = np.multiply(5., data1) + np.random.uniform(low=0., high=100., size=step_num)
    # data3 = np.multiply(10., data1) - np.random.uniform(low=0., high=100., size=step_num)
    # X2D = np.stack([data1, data2, data3])
    # X2D = np.transpose(X2D)

    # Generate dummy 3D data:
    X3D = [[[1, 2], [10, 20], [100, 200]],
           [[3, 4], [30, 40], [300, 400]],
           [[5, 6], [50, 60], [500, 600]]]
    # print(X3D)

    timestep_number = len(X3D)
    # print(timestep_number)
    particle_number = len(X3D[0])
    # print(particle_number)
    dim_number = len(X3D[0][0])
    # print(dim_number)

    # Convert 3D data to 2D data
    X2D = np.zeros((dim_number*timestep_number, particle_number))
    for i in range(timestep_number):
        for j in range(particle_number):
            for k in range(dim_number):
                X2D[dim_number*i+k][j] = X3D[i][j][k]
    # print(X2D)

    idx, W, Y2D = pca_evd_np(X2D, MODE_NUMBER)
    # print(Y2D)

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
    print(Y2D_2)

    # plot(X2D, Y2D, 2*timestep_number)

    ## Test1:
    # G = [1, 2, 10, 20, 100, 200, 3, 4, 30, 40, 300, 400, 5, 6, 50, 60, 500, 600]
    # position_shape = [3, -1, 2]
    # parsed_features = np.reshape(G, position_shape)
    # print(parsed_features)

    ## Test2:
    import struct
    print(struct.pack('q', 6))
    print(struct.pack('d', 612.1312211))
    a = [612.1312211, 612.1312211]
    bufa = struct.pack('%sd' % len(a), *a)
    print(bufa)
    b = [6, 6]
    bufb = struct.pack('%sq' % len(b), *b)
    print(bufb)
