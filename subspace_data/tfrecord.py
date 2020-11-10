import os
import json
import tensorflow.compat.v1 as tf

from learning_to_simulate import reading_utils


def deserialize(dataset_name, split):
    # Loads the metadata of the dataset.
    data_path = './learning_to_simulate/datasets/' + dataset_name
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

    return X3D