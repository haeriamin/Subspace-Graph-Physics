'''
Example usage (from parent directory):
`python -m pca.preprocess_tfrecord`

Refs:
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
https://aihub.cloud.google.com/u/0/p/products%2Fcd31a10b-85db-4c78-830c-a3794dd606ce
https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

'''

import os
import json
import numpy as np
import tensorflow.compat.v1 as tf

from learning_to_simulate import reading_utils
from pca import apply_pca


if __name__ == '__main__':

    # either 'train', 'valid' or 'test'.
    split = 'train'

    # Loads the metadata of the dataset.
    data_path = './learning_to_simulate/datasets/Sand'
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        metadata = json.loads(fp.read())

    # Create a tf.data.Dataset from the TFRecord.
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(
            [os.path.join(data_path, f'{split}.tfrecord')])
    _, serialized_example = reader.read(filename_queue)
    context_data, sequence_data = reading_utils.parse_serialized_simulation_example(
            serialized_example, metadata)

    # Many tf.train functions use tf.train.QueueRunner,
    # so we need to start it before we read
    tf.train.start_queue_runners(tf.InteractiveSession())

    # Context:
    particle_type_array = []
    for name, tensor in context_data.items():
        particle_type_array.append(tensor.eval())
        # print('{}: {}'.format(name, tensor.eval()))
    particle_type = particle_type_array[0].tolist()
    # particle_number = len(particle_type)

    # Feature:
    particle_pos_array = []
    for name, tensor in sequence_data.items():
        particle_pos_array.append(tensor.eval())
        # print('{}: {}'.format(name, tensor.eval()))
    X3D = particle_pos_array[0].tolist()

    timestep_number = len(X3D)
    # print(timestep_number)
    particle_number = len(X3D[0])
    # print(particle_number)
    dim_number = len(X3D[0][0])
    # print(dim_number)

    # Start preprocessing data
    MODE_NUMBER = 256

    # Convert 3D data to 2D data
    X2D = np.zeros((2*timestep_number, particle_number))
    for i in range(timestep_number):
        for j in range(particle_number):
            for k in range(dim_number):
                X2D[2*i+k][j] = X3D[i][j][k]

    # Apply PCA to 2D data
    id, W, Y2D = apply_pca.pca_evd_np(X2D, MODE_NUMBER)

    # Convert 2D data to 3D data
    Y3D = np.zeros((timestep_number, MODE_NUMBER, dim_number))
    for i in range(2*timestep_number):
        for j in range(MODE_NUMBER):
            Y3D[i // 2][j][i % 2] = Y2D[i][j]

    print(id)
           