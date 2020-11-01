'''Example usage (from parent directory):
`python -m learning_to_simulate.pca --data_path={DATA_PATH}`

'''
import functools
import json
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from learning_to_simulate import train
from learning_to_simulate import reading_utils


# x = tf.random.uniform([3, 3])
# print("Is there a GPU available? "),
# print(tf.config.experimental.list_physical_devices("GPU"))
# print("Is the Tensor on GPU #0?  "),
# print(x.device.endswith('GPU:0'))

# added: PCA mode number 
MODE_NUMBER = 5

# So we can calculate the last 5 velocities.
INPUT_SEQUENCE_LENGTH = 6
# either 'train', 'valid' or 'test'.
split = 'test'

# Loads the metadata of the dataset.
data_path = './learning_to_simulate/datasets/Sand'
with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    metadata = json.loads(fp.read())

# Create a tf.data.Dataset from the TFRecord.
ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
ds = ds.map(functools.partial(
    reading_utils.parse_serialized_simulation_example, metadata=metadata))
# print(ds)

# added:
pca = functools.partial(reading_utils.apply_pca, mode_number=MODE_NUMBER)
ds = ds.map(pca)
# print(ds)


# -----------------------------------------------------------------------------


# MODE_NUMBER = 3
# particle_num = 3
# step_num = 100

# data1 = tf.random.uniform([step_num], minval=0., maxval=100., dtype=tf.float32, seed = 1)
# data2 = tf.multiply(5., data1) + tf.random.uniform([step_num], minval=1., maxval=100., dtype=tf.float32, seed = 0)
# data3 = tf.multiply(10., data1) - tf.random.uniform([step_num], minval=1., maxval=100., dtype=tf.float32, seed = 2)
# X = tf.stack([data1, data2, data3])
# X = tf.transpose(X)

# with tf.Session() as sess:
#     Xnp = sess.run(X)
# means = np.mean(Xnp, axis=0)
# Xn = Xnp
# for i in range(particle_num):
#     Xn[:,i] = Xnp[:,i] - means[i]

# Xtf = tf.placeholder(tf.float32, shape=[step_num, particle_num])
# Covtf = tf.placeholder(tf.float32, shape=[particle_num, particle_num])

# Cov = np.matmul(np.transpose(Xn),Xn)
# stf, utf, vtf = tf.linalg.svd(Covtf, full_matrices=True)

# Wtf = tf.strided_slice(vtf, begin=[0, 0], end=[particle_num, MODE_NUMBER])

# Ytf = tf.matmul(Xtf, Wtf)

# # dependencies = reading_utils.get_tensor_dependencies(Ytf)
# # print(dependencies)

# with tf.Session() as sess:
#     Yn = sess.run(Ytf, feed_dict = {Xtf: Xn, Covtf: Cov})
#     s = sess.run(stf, feed_dict = {Xtf: Xn, Covtf: Cov})
#     v = sess.run(vtf, feed_dict = {Xtf: Xn, Covtf: Cov})

# # print(s)
# # print(v)

# fig = plt.figure()
# I = range(step_num)
# plt.plot(I, Xn[:,0], '.', color='b')
# plt.plot(I, Xn[:,1], '.', color='r')
# plt.plot(I, Xn[:,2], '.', color='k')

# plt.plot(I, Yn[:,0], '+', color='k')
# plt.plot(I, Yn[:,1], 'x', color='r')
# plt.plot(I, Yn[:,2], '|', color='b')

# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(Xn[:,0],Xn[:,1],Xn[:,2], marker='.')
# # ax.scatter(Yn[:,0],Yn[:,1],Yn[:,2], marker='+')
# plt.show()


# -----------------------------------------------------------------------------


## APPLY PCA BEFORE THESE STEPS
# # Split an entire trajectory into chunks of 7 steps.
# # Previous 5 velocities, current velocity and target.
# split_with_window = functools.partial(
#     reading_utils.split_trajectory,
#     window_length=INPUT_SEQUENCE_LENGTH+1)
# ds = ds.flat_map(split_with_window)

# # Split a chunk into input steps and target steps
# ds = ds.map(train.prepare_inputs)


# -----------------------------------------------------------------------------


## OPTION 0:
# mean = tf.reduce_mean(X, axis=0)  # calculates the mean
# X_n = X - mean
# eigen_values, eigen_vectors = tf.linalg.eigh(tf.tensordot(tf.transpose(X_n), X_n, axes=1))
# w = tf.slice(eigen_vectors, [0, 0], [X.shape[1], 2])
# X_sub = tf.tensordot(X_n, w, axes=1)
# X_sub += mean
# with tf.Session() as sess:
#     X = sess.run(X)
#     X_sub = sess.run(X_sub)
#     eigen_values = sess.run(eigen_values)
#     eigen_vectors = sess.run(eigen_vectors)
#     w = sess.run(w)
# print(eigen_values)
# print(eigen_vectors)
# print(w)

## OPTION 1:
# with tf.Session() as sess:
#     Xnp = sess.run(X)
# means = np.mean(Xnp, axis=0)
# X_n = np.subtract(Xnp, means)
# cov_mat = np.cov(X_n)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print(eigen_vals)
# print(eigen_vecs)
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# print(w)
# X_sub = X_n.dot(w)
# # Xtf_sub = tf.placeholder(tf.float32, shape=[Xn.shape[0], Xn.shape[1]])
# # Xtf_sub = X_sub
# # with tf.Session() as sess:
# #     Xtf_sub = sess.run(Xtf_sub, feed_dict = {X_sub: X_sub})