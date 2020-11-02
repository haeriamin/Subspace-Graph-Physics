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


def apply_pca(context, features):
    mode_number = 3


    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features['position'], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]
    # Remove the target from the input.
    out_dict['position'] = pos[:, :-1]
    # Compute the number of nodes
    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict, target_position


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

    fig = plt.figure()
    I = range(step_num)
    plt.plot(I, Xn_val[:,0], '.', color='b')
    plt.plot(I, Xn_val[:,1], '.', color='r')
    plt.plot(I, Xn_val[:,2], '.', color='k')

    plt.plot(I, Yn_val[:,0], '+', color='k')
    plt.plot(I, Yn_val[:,1], 'x', color='r')
    plt.plot(I, Yn_val[:,2], '|', color='b')

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Xn_val[:,0],Xn_val[:,1],Xn_val[:,2], marker='.')
    # ax.scatter(Yn_val[:,0],Yn_val[:,1],Yn_val[:,2], marker='+')
    plt.show()


# Just for testing
if __name__ == '__main__':

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

    # added:
    # pca = functools.partial(reading_utils.apply_pca, mode_number=MODE_NUMBER)
    # ds = ds.map(pca)
    # print(ds)

    ds = ds.map(apply_pca)
