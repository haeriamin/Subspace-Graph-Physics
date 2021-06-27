'''
Refs:
https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
https://medium.com/ymedialabs-innovation/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af

https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
https://aihub.cloud.google.com/u/0/p/products%2Fcd31a10b-85db-4c78-830c-a3794dd606ce

'''

import os
import json
import tensorflow.compat.v1 as tf
import functools
import pandas as pd

from learning_to_simulate import reading_utils


def deserialize(data_path, split):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        metadata = json.loads(fp.read())

    ds = tf.data.TFRecordDataset(
        [os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        reading_utils.parse_serialized_simulation_example, metadata=metadata))
    iterator = ds.make_one_shot_iterator()
    c = 0
    particle_type = []
    particle_key = []
    particle_pos = []
    with tf.Session() as sess:
        next_element = iterator.get_next()
        while True:
            try:
                context_data, sequence_data = sess.run(next_element)
                # context_data, sequence_data, t1, t2 = sess.run(next_element)

                # Context:
                particle_type.append(context_data['particle_type'].tolist())
                particle_key.append(context_data['key'].tolist())

                # Feature:
                particle_pos.append(sequence_data['position'].tolist())

                c += 1
                print(c)

                # Just for testing
                if c == 5:
                    break

            except tf.errors.OutOfRangeError:
                break

    # print(t1)
    # print(len(t1))
    # print(len(t1[0]))
    # print(len(t1[0][0]))
    # print('---')
    # # print(t2)
    # print(len(t2))
    # print(len(t2[0]))
    # print(len(t2[0][0]))
    # print('---')
    return particle_type, particle_key, particle_pos


def _serialize(data):
    # Create context
    context = tf.train.Features(feature={
        'particle_type': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data['context1']])),
                # value=[m for m in data['context1']])),

        'key': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[data['context2']])),
    })

    # Create sequence data
    # Particle position
    position_feat = []
    for i in range(len(data['feature1'])):
        pos_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data['feature1'][i]]))
        position_feat.append(pos_feature)
    position = tf.train.FeatureList(feature=position_feat)

    # Rigid body force
    force_feat = []
    for i in range(len(data['feature2'])):
        force_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data['feature2'][i]]))
        force_feat.append(force_feature)
    force = tf.train.FeatureList(feature=force_feat)

    feature = tf.train.FeatureLists(feature_list={
        'position': position,
        'force': force,
    })

    # Create the SequenceExample
    example = tf.train.SequenceExample(
        context=context, feature_lists=feature)

    return example


def serialize(data_path, split, data):
    filename = os.path.join(data_path, f'{split}.tfrecord')
    example = _serialize(data)
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def read_csv(data_path):
    data = pd.read_csv(data_path, header=None, usecols=[0, 1, 2, 3, 4],
                       names=['col0', 'col1', 'col2', 'col3', 'col4'])
    return data


def write_json(data_path, bounds, frames, particles, dimension, frame_rate,
               particle_vel_mean, particle_vel_std,
               particle_acc_mean, particle_acc_std,
               rigid_body_force_mean, rigid_body_force_std):
    data = {}
    data['bounds'] = bounds
    data['sequence_length'] = frames
    data['default_connectivity_radius'] = 0.025
    data['dim'] = dimension
    data['dt'] = round(1/frame_rate, 6)
    data['vel_mean'] = particle_vel_mean
    data['vel_std'] = particle_vel_std
    data['acc_mean'] = particle_acc_mean
    data['acc_std'] = particle_acc_std
    data['particle_number'] = particles  # added
    data['rigid_body_force_mean'] = rigid_body_force_mean  # added
    data['rigid_body_force_std'] = rigid_body_force_std  # added

    with open(os.path.join(data_path, 'metadata.json'), 'w') as outfile:
        json.dump(data, outfile)
    
    del data
