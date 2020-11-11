'''
Refs:
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
https://aihub.cloud.google.com/u/0/p/products%2Fcd31a10b-85db-4c78-830c-a3794dd606ce

'''

import os
import json
import tensorflow.compat.v1 as tf
import functools

from learning_to_simulate import reading_utils


def deserialize(name, split):
    # Loads the metadata of the dataset.
    data_path = './learning_to_simulate/datasets/' + name
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        metadata = json.loads(fp.read())

    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        reading_utils.parse_serialized_simulation_example, metadata=metadata))
    iterator = ds.make_one_shot_iterator()
    with tf.Session() as sess:
        context_data, sequence_data = sess.run(iterator.get_next())
        # context_data, sequence_data, t1, t2 = sess.run(iterator.get_next())

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
    
    # Context:
    particle_type = context_data['particle_type'].tolist()
    particle_key = context_data['key'].tolist()

    # Feature:
    particle_pos = sequence_data['position'].tolist()

    return particle_type, particle_key, particle_pos


def serialize(data, name, split):
    # Loads the metadata of the dataset.
    data_path = './learning_to_simulate/datasets/' + name
    # with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    #     metadata = json.loads(fp.read())

    # Create context
    context = tf.train.Features(feature={
        'particle_type': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[data['context1']])),
                # value=[m for m in data['context1']])),
        'key': tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=[data['context2']])),
    })

    # Create sequence data
    position_feat = []
    for i in range(len(data['feature'])):
        pos_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[data['feature'][i]]))
        position_feat.append(pos_feature)
    position = tf.train.FeatureList(feature=position_feat)
    feature = tf.train.FeatureLists(feature_list={
        'position': position,
    })

    # Create the SequenceExample
    example = tf.train.SequenceExample(context=context, feature_lists=feature)

    # Write tfrecord file
    filename = os.path.join(data_path, f'{split}.tfrecord')
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())
