'''
Example usage (from parent directory):
'python -m pca.test_pca_svd_tf'

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


def apply_pca(X, mode_number):
    particle_num = X.get_shape()[1]
    step_num = X.get_shape()[0]

    means = tf.reduce_mean(X, axis=0)
    Xn = tf.math.subtract(X, means)

    S, U, V = tf.linalg.svd(Xn, full_matrices=True)

    W = tf.strided_slice(V, begin=[0, 0], end=[particle_num, mode_number])

    Yn = tf.matmul(Xn, W)

    with tf.Session() as sess:
        Yn_val = sess.run(Yn)
        Xn_val = sess.run(Xn)
        S_val = sess.run(S)
        V_val = sess.run(V)

    print(S_val)
    print(V_val)

    I = range(step_num)
    # plt.plot(I, Xn_val[:, 0], '.', color='b')
    # plt.plot(I, Xn_val[:, 1], '.', color='r')
    # plt.plot(I, Xn_val[:, 2], '.', color='k')
    # plt.plot(I, Yn_val[:, 0], '+', color='k')
    # plt.plot(I, Yn_val[:, 1], 'x', color='r')
    # plt.plot(I, Yn_val[:, 2], '|', color='b')
    # plt.xlabel("Timestep")
    # plt.ylabel("Position")
    # plt.legend(['Full1', 'Full2', 'Full3', 'Sub1', 'Sub2', 'Sub3'])

    def axisEqual3D(ax):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xn_val[:,0],Xn_val[:,1],Xn_val[:,2], marker='.')
    ax.scatter(Yn_val[:,2],Yn_val[:,1],Yn_val[:,0], marker='+')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylabel("Z")
    axisEqual3D(ax)
    plt.legend(['Full', 'Sub'])

    plt.savefig('pca_test.png', dpi=1000)
    plt.show()


# Just for testing
if __name__ == '__main__':

    step_num = 100

    # Generate dummy data
    data1 = tf.random.uniform([step_num], minval=0., maxval=100.,
        dtype=tf.float32, seed=1)
    data2 = tf.multiply(5., data1) + tf.random.uniform([step_num], minval=1.,
        maxval=100., dtype=tf.float32, seed=0)
    data3 = tf.multiply(10., data1) - tf.random.uniform([step_num], minval=1.,
        maxval=100., dtype=tf.float32, seed=2)
    X = tf.stack([data1, data2, data3])
    X = tf.transpose(X)

    MODE_NUMBER = 3
    apply_pca(X, MODE_NUMBER)