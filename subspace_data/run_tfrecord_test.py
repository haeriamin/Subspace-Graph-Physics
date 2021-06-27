'''
Example usage (from parent directory):
`python -m subspace_data.run_tfrecord_test`

Ref:
https://docs.python.org/3/library/struct.html

'''

import os
import numpy as np
import struct
import math

from subspace_data import tfrecord


if __name__ == '__main__':

    # If dimensional analysis used
    scale_dim = 0.25
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2

    # Simulation properties
    dataset = 'Excavation'
    dataset_split = 'train'  # 'train', 'valid' or 'test'
    frame_rate = 120 * scale_time
    dimension = 3
    bounds = [[0.025/scale_dim, (0.025+0.2)/scale_dim],
              [0.025/scale_dim, (0.025+0.2)/scale_dim],
              [0.025/scale_dim, (0.025+0.15)/scale_dim]]

    c = 0
    particle_key = 0
    dataset = './learning_to_simulate/datasets/' + dataset

    # for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:

    #     for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:

    #         for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:

    #             for motion in [1.0, 2.0]:

    # Just for test
    for co in range(2):

        # task_id = str(c) + '_D' + str(depth) + 'm_S' + str(
        #     speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)

        # Just for test
        if co == 1:
            task_id = 'D0.1m_S0.1ms-1_A45.0deg_M2.0'
        else:
            task_id = 'D0.1m_S0.15ms-1_A10.0deg_M1.0'

        c += 1
        ds_path = os.path.join(dataset, task_id)

        path, dirs, files = next(os.walk(os.path.join(ds_path, 'frames')))
        frames = len(files)-1

        # list_of_tfrecord_files = []
        # segments = math.ceil(frames/1000)
        # for segment in range(segments):
        particle_type = []
        particle_pos = []
        particle_vel = []
        particle_acc = []
        rigid_body_force = []
        print(task_id)
        for frame in range(1, frames+1):
        # for particle in range(segment*1000, min((segment+1)*1000, frames):
            file_path = os.path.join(
                ds_path, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')

            data_frame = tfrecord.read_csv(file_path)
            particles = len(data_frame)

            particle_pos_frame = []
            particle_vel_frame = []
            particle_acc_frame = []
            rigid_body_force_frame = []
            for particle in range(1, particles):
                # Particle type
                if frame == 1:
                    particle_type.append(  # soil: 6, rigid: 3
                        -3 * int(data_frame['col0'][particle]) + 6)

                # Particle position at each frame
                particle_pos_frame.append([
                    round(data_frame['col1'][particle]/scale_dim, 6),
                    round(data_frame['col2'][particle]/scale_dim, 6),
                    round(data_frame['col3'][particle]/scale_dim, 6)])

            # Rigid body force
            rigid_body_force.append([
                round(data_frame['col0'][0]/(scale_dim**3), 6),
                round(data_frame['col1'][0]/(scale_dim**3), 6),
                round(data_frame['col2'][0]/(scale_dim**3), 6)])

            # Particle position
            particle_pos.append(particle_pos_frame)

            # Extra frame in the beginning
            if frame == 1:
                particle_pos.append(particle_pos_frame)
                rigid_body_force.append([
                    round(data_frame['col0'][0]/(scale_dim**3), 6),
                    round(data_frame['col1'][0]/(scale_dim**3), 6),
                    round(data_frame['col2'][0]/(scale_dim**3), 6)])
            del particle_pos_frame
            del data_frame
            print(frame)

        particles = len(particle_pos[0])

        # Mean and std deviation of rigid body forces
        rigid_body_force_mean = np.mean(rigid_body_force, axis=0).tolist()
        rigid_body_force_std = np.std(rigid_body_force, axis=0).tolist()

        ''' Serialize via tfrecord '''
        particle_type_bytes = struct.pack(
            '%sq' % len(particle_type), *particle_type)
        del particle_type

        # Convert 3D data to 2D data
        particle_pos_2D = np.zeros(
            (frames+1, particles*dimension))
        for i in range(frames+1):
            for j in range(particles):
                for k in range(dimension):
                    particle_pos_2D[i][dimension*j+k] = particle_pos[i][j][k]

        particle_pos_bytes = []
        rigid_body_force_bytes = []
        for i in range(frames+1):
            particle_pos_bytes.append(struct.pack(
                '%sf' % len(particle_pos_2D[i]), *particle_pos_2D[i]))
            rigid_body_force_bytes.append(struct.pack(
                '%sf' % len(rigid_body_force[i]), *rigid_body_force[i]))
        del particle_pos_2D
        del rigid_body_force

        data = {
            'context1': particle_type_bytes,
            'context2': particle_key,
            'feature1': particle_pos_bytes,
            'feature2': rigid_body_force_bytes
        }
        del particle_type_bytes
        del particle_pos_bytes
        del rigid_body_force_bytes

        tfrecord.serialize(ds_path, dataset_split, data)

        ''' Create metadata json file '''
        # In a separate loop to avoid OOM issue
        for frame in range(1, frames+1):
            # Particle velocity
            nparray = (np.array(particle_pos[frame]) -
                       np.array(particle_pos[frame-1])) * frame_rate
            particle_vel.append(nparray.tolist())

            # Particle acceleration
            if frame >= 2:
                nparray = (np.array(particle_vel[frame-1]) -
                           np.array(particle_vel[frame-2])) * frame_rate
                particle_acc.append(nparray.tolist())
        del particle_pos

        # Mean and std deviation of particle velocities
        aux = np.mean(particle_vel, axis=1).tolist()
        particle_vel_mean = np.mean(aux, axis=0).tolist()
        aux = np.mean(particle_vel, axis=1).tolist()
        particle_vel_std = np.std(aux, axis=0).tolist()
        del particle_vel

        # Mean and std deviation of particle accelerations
        aux = np.mean(particle_acc, axis=1).tolist()
        particle_acc_mean = np.mean(aux, axis=0).tolist()
        aux = np.mean(particle_acc, axis=1).tolist()
        particle_acc_std = np.std(aux, axis=0).tolist()
        del particle_acc

        tfrecord.write_json(
            ds_path,
            bounds, frames, particles, dimension, frame_rate,
            particle_vel_mean, particle_vel_std,
            particle_acc_mean, particle_acc_std,
            rigid_body_force_mean, rigid_body_force_std)

        # list_of_tfrecord_files.append()

    # # Create dataset from multiple .tfrecord files
    # dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)
    # filename = 'train.tfrecord'
    # writer = tf.data.experimental.TFRecordWriter(filename)
    # writer.write(dataset)
