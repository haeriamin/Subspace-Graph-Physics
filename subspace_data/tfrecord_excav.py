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
import pandas as pd


def serialize(data):
    ## Create context
    context = tf.train.Features(feature={
        'particle_type': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data['context1']])),
        'key': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[data['context2']])),
    })

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

    ## Create features
    feature = tf.train.FeatureLists(feature_list={
        'position': position,
        'force': force,
    })

    # Create the SequenceExample
    example = tf.train.SequenceExample(
        context=context,
        feature_lists=feature,
        )

    return example


# def serialize(data_path, split, data):
#     filename = os.path.join(data_path, f'{split}.tfrecord')
#     example = _serialize(data)
#     with tf.python_io.TFRecordWriter(filename) as writer:
#         writer.write(example.SerializeToString())


def read_csv(data_path):
    data = pd.read_csv(data_path, header=None, usecols=[0, 1, 2, 3, 4],
                       names=['col0', 'col1', 'col2', 'col3', 'col4'])
    return data


def write_json(data_path, bounds, frames, dimension, frame_rate,
               particle_vel_mean, particle_vel_std,
               particle_acc_mean, particle_acc_std,
               rigid_body_force_mean, rigid_body_force_std, dataset_split,
               rigid_body_ref_point=None, particles_rigid=None, mode_number=None):
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
    # added:
    data['rigid_body_force_mean'] = rigid_body_force_mean
    data['rigid_body_force_std'] = rigid_body_force_std
    data['rigid_body_ref_point'] = rigid_body_ref_point
    data['particles_rigid'] = particles_rigid
    data['mode_number'] = mode_number
    data['dataset_split'] = dataset_split

    with open(os.path.join(data_path, 'metadata.json'), 'w') as outfile:
        json.dump(data, outfile)
