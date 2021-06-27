'''
Example usage (from parent directory):
`python -m subspace_data.run_tfrecord_excav_pca`

'''

# import pickle
import os
import struct
import pickle
import random
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from subspace_data import tfrecord_excav
from subspace_data import pca_plot
from subspace_data import pca


MODE_NUMBER = 8
VISUALIZATION = False


if __name__ == '__main__':
    dataset_load = './learning_to_simulate/datasets/' + 'Excavation'
    dataset_save = dataset_load + '_PCA'


    # Shuffle and create train, valid, and test examples (cases)
    no_examples = 5*5*5*2  # 250
    no_train_examples = int(0.90 * no_examples)  # to train model
    no_valid_examples = int(0.05 * no_examples)  # to tune hyperparameters  
    no_test_examples  = int(0.05 * no_examples)  # to evaluate final model

    example_list = list(range(1, 1 + no_examples))
    random.shuffle(example_list)
    train_example_list = random.choices(example_list, k=no_train_examples)
    example_list = [ elem for elem in example_list if elem not in train_example_list]
    valid_example_list = random.choices(example_list, k=no_valid_examples)
    example_list = [ elem for elem in example_list if elem not in valid_example_list]
    test_example_list = random.choices(example_list, k=no_test_examples)

    # train_example_list = [240]
    dataset_split = {
        'train': train_example_list,
        'valid': valid_example_list,
        'test': test_example_list
    }

    # Exclude bad examples (cases)
    for split in dataset_split:
        c1 = 0
        for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
            for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
                for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                    for motion in [1.0, 2.0]:
                        c1 += 1
                        if speed == 0.01:  
                            dataset_split[split] = [ elem for elem in dataset_split[split] if elem != c1]
        print(dataset_split[split])


    # Load PCA loading matrix and mean scalar
    with open(dataset_save + '/' + 'pca_eigen_vectors.pkl', 'rb') as f:
        W = pickle.load(f)
        W = W[:, :MODE_NUMBER]
    with open(dataset_save + '/' + 'pca_mean_scalar.pkl', 'rb') as f:
        MEAN = pickle.load(f)


    # Simulation properties
    scale_dim = 0.25  # If dimensional analysis used
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2
    frame_rate = 120 * scale_time
    dimension = 3
    bounds = [[0.1, 0.9],
              [0.2, 1.0],
              [0.1, 0.9]]
    plate_height = 0.0763/scale_dim
    plate_width = 0.1142/scale_dim


    # Start
    for counter, split in enumerate(dataset_split):
        # Initialization 
        particle_key = 0
        c2, c3, frames = 0, 0, 0
        if split == 'train':   
            particle_vel_mean, particle_vel_var = [0., 0., 0.], [0., 0., 0.]
            particle_acc_mean, particle_acc_var = [0., 0., 0.], [0., 0., 0.]
            rigid_body_force_mean, rigid_body_force_var = [0., 0., 0.], [0., 0., 0.]
            n1_v, n2_v = 0, 0
            n1_a, n2_a = 0, 0
            n1_f, n2_f = 0, 0

        tfr_filename = os.path.join(dataset_save, f'{split}.tfrecord')
        with tf.python_io.TFRecordWriter(tfr_filename) as writer:
            
            len_split = len(dataset_split[split])
            while c2 < len_split:

                c1 = 0
                for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
                    for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
                        for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                            for motion in [1.0, 2.0]:
                                c1 += 1

                                if dataset_split[split][c3] == c1:

                                    task_id = str(c1) + '_D' + str(depth) + 'm_S' + str(
                                        speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)

                                    ds_path_load = os.path.join(dataset_load, task_id)
                                    if VISUALIZATION:
                                        ds_path_save = os.path.join(dataset_save, task_id)
                                        if not os.path.exists(ds_path_save):
                                            os.makedirs(ds_path_save)

                                    path, dirs, files = next(os.walk(os.path.join(ds_path_load, 'frames')))
                                    frames = min(320, len(files)-1)

                                    particle_id = []
                                    particle_type = []
                                    particle_pos = []
                                    rigid_body_force = []
                                    if VISUALIZATION:
                                        particle_type_np = []
                                        particle_pos_np = []
                                    print(task_id)
                                    for frame in range(1, frames+1):
                                        r_particles = 0
                                        particle_id_frame = []
                                        particle_type_frame = []
                                        particle_pos_frame = []
                                        rigid_body_force_frame = []
                                        r_particle_pos_x_min = 10
                                        r_particle_pos_y_min = 10
                                        r_particle_pos_z_min = 10
                                        if VISUALIZATION:
                                            particle_type_frame_np = []
                                            particle_pos_frame_np = []

                                        file_path = os.path.join(
                                            ds_path_load, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')

                                        data_frame = tfrecord_excav.read_csv(file_path)
                                        all_particles = len(data_frame)

                                        # Get one particle in rigid body
                                        for particle in reversed(range(1, all_particles)):
                                            if int(data_frame['col0'][particle]) == 1:
                                                minx = round(float(data_frame['col1'][particle])/scale_dim, 6)
                                                miny = round(float(data_frame['col2'][particle])/scale_dim, 6)
                                                minz = round(float(data_frame['col3'][particle])/scale_dim, 6)
                                                if minx < r_particle_pos_x_min:
                                                    r_particle_pos_x_min = minx
                                                if miny < r_particle_pos_y_min:
                                                    r_particle_pos_y_min = miny
                                                if minz < r_particle_pos_z_min:
                                                    r_particle_pos_z_min = minz
                                        # Create structured rigid body
                                        res = 10000
                                        grid_size = int(res*0.10)  # [m]
                                        for iz in range(int(r_particle_pos_z_min*res), int((r_particle_pos_z_min+plate_width)*res), grid_size):
                                            for iy in range(int(r_particle_pos_y_min*res), int((r_particle_pos_y_min + plate_height*np.cos(np.deg2rad(angle)))*res), grid_size):
                                                r_particles += 1
                                                particle_id_frame.append(-1)
                                                particle_pos_frame.append([
                                                    r_particle_pos_x_min + np.tan(np.deg2rad(angle))*(iy/res-r_particle_pos_y_min),
                                                    iy/res,
                                                    iz/res])
                                                # Particle type: soil: 6, boundary: 3, rigid: 0
                                                particle_type_frame.append(3)
                                                if VISUALIZATION:
                                                    particle_pos_frame_np.append(np.array([
                                                        r_particle_pos_x_min + np.tan(np.deg2rad(angle))*(iy/res-r_particle_pos_y_min),
                                                        iy/res,
                                                        iz/res]))
                                                    particle_type_frame_np.append(np.array(3))

                                        for particle in range(1, all_particles):
                                            if ~np.isnan(float(data_frame['col1'][particle])):
                                                if (int(data_frame['col0'][particle]) == 0) and (particle%5 == 0):
                                                    particle_id_frame.append(particle)
                                                    particle_pos_frame.append([
                                                        round(float(data_frame['col1'][particle])/scale_dim, 6),
                                                        round(float(data_frame['col2'][particle])/scale_dim, 6),
                                                        round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                                    particle_type_frame.append(6)
                                                    if VISUALIZATION:
                                                        particle_pos_frame_np.append(np.array([
                                                            round(float(data_frame['col1'][particle])/scale_dim, 6),
                                                            round(float(data_frame['col2'][particle])/scale_dim, 6),
                                                            round(float(data_frame['col3'][particle])/scale_dim, 6)]))
                                                        particle_type_frame_np.append(np.array(6))
                                            else:
                                                particle_id_frame.append(particle)
                                                particle_type_frame_np.append(np.array(6))
                                                particle_pos_frame.append(particle_pos[frame-1][particle])

                                        particle_id.append(particle_id_frame)
                                        particle_type.append(particle_type_frame)
                                        particle_pos.append(particle_pos_frame)
                                        rigid_body_force.append([
                                            round(float(data_frame['col0'][0])/(scale_dim**3), 6),
                                            round(float(data_frame['col1'][0])/(scale_dim**3), 6),
                                            round(float(data_frame['col2'][0])/(scale_dim**3), 6)])
                                        if VISUALIZATION:
                                            particle_type_np.append(np.array(particle_type_frame_np))
                                            particle_pos_np.append(np.array(particle_pos_frame_np))
                                        # Extra frame in the beginning
                                        if frame == 1:
                                            particle_id.append(particle_id_frame.copy())
                                            particle_type.append(particle_type_frame.copy())
                                            particle_pos.append(particle_pos_frame.copy())
                                            rigid_body_force.append([
                                                round(float(data_frame['col0'][0])/(scale_dim**3), 6),
                                                round(float(data_frame['col1'][0])/(scale_dim**3), 6),
                                                round(float(data_frame['col2'][0])/(scale_dim**3), 6)])
                                            if VISUALIZATION:
                                                particle_type_np.append(np.array(particle_type_frame_np))
                                                particle_pos_np.append(np.array(particle_pos_frame_np))
                                        if frame % 10 == 0:
                                            print(c2+1, ':', round(frame/frames, 3)*100, '%')
                                    del particle_pos_frame
                                    del data_frame
                                    particles = len(particle_pos[0])


                                    ''' Process data '''
                                    # print(r_particles)
                                    # Convert 3D soil data to 2D data
                                    X = np.zeros((dimension*(frames+1), particles-r_particles))
                                    for i in range((frames+1)):
                                        for j in range(r_particles, particles):
                                            for k in range(dimension):
                                                X[k+i*dimension][j-r_particles] = particle_pos[i][j][k] - MEAN

                                    # Apply PCA to 2D data
                                    Xr2D = np.matmul(X, W)

                                    # Convert 2D data to 3D data
                                    Xr = np.zeros((frames+1, MODE_NUMBER+r_particles, dimension))
                                    for i in range(frames+1):
                                        for j in range(r_particles):
                                            for k in range(dimension):
                                                Xr[i][j][k] = particle_pos[i][j][k]
                                    for i in range(dimension*(frames+1)):
                                        for j in range(MODE_NUMBER):
                                            Xr[i // int(dimension)][j+r_particles][i % int(dimension)] = Xr2D[i][j]

                                    if VISUALIZATION:
                                        particle_type_np_r = np.ones(((frames+1), r_particles+MODE_NUMBER))*3
                                        for i in range(frames+1):
                                            for j in range(r_particles, r_particles+MODE_NUMBER):
                                                particle_type_np_r[i][j] = particle_type_np[i][j]
                                        pca_plot.plot(Xr, particle_type_np_r)
                                        del particle_type_frame_np
                                        del particle_pos_frame_np
                                        del particle_pos_np
                                        del particle_type_np


                                    ''' Serialize via tfrecord '''
                                    # Convert 3D data to 2D data
                                    particle_pos_2D = np.zeros((frames+1, (r_particles+MODE_NUMBER)*dimension))
                                    for i in range(frames+1):
                                        for j in range(r_particles+MODE_NUMBER):
                                            for k in range(dimension):
                                                particle_pos_2D[i][dimension*j+k] = Xr[i][j][k]

                                    particle_type_bytes = []
                                    particle_pos_bytes = []
                                    rigid_body_force_bytes = []

                                    # ind = np.argmax(particles)
                                    particle_type_bytes = struct.pack(
                                            '%sq' % len(particle_type[0][:r_particles+MODE_NUMBER]), *particle_type[0][:r_particles+MODE_NUMBER])

                                    for i in range(frames+1):
                                        particle_pos_bytes.append(struct.pack(
                                            '%sf' % len(particle_pos_2D[i]), *particle_pos_2D[i]))

                                        rigid_body_force_bytes.append(struct.pack(
                                            '%sf' % len(rigid_body_force[i]), *rigid_body_force[i]))

                                    data = {
                                        'context1': particle_type_bytes,
                                        'context2': particle_key,
                                        'feature1': particle_pos_bytes,
                                        'feature2': rigid_body_force_bytes,
                                    }
                                    del particle_type
                                    del particle_pos_2D
                                    del particle_type_bytes
                                    del particle_pos_bytes
                                    del rigid_body_force_bytes
                                    del particle_pos

                                    example = tfrecord_excav.serialize(data)
                                    writer.write(example.SerializeToString())


                                    ''' Calculate mean and variance '''
                                    if split == 'train':
                                        # # Calculate minimum particle distance to each particle => ~2cm => R=0.03m is the best
                                        # for i in range(len(Xr[0])):
                                        #     dist_i = 1000
                                        #     for j in range(len(Xr[0])):
                                        #         if i != j:
                                        #             dist_j = np.linalg.norm(np.array(Xr[-1][j])-np.array(Xr[-1][i]))
                                        #             if dist_j < dist_i:
                                        #                 dist_i = dist_j
                                        #     print(dist_i*100)

                                        # Calculate particle velocities and accelerations
                                        cc = 1
                                        particle_vel = []
                                        particle_acc = []
                                        for frame in range(1, frames):
                                            # Particle velocity (excluding rigid particles)
                                            nparray1 = np.subtract(
                                                np.array(Xr[frame][r_particles:]),
                                                np.array(Xr[frame-1][r_particles:])) #* frame_rate
                                            particle_vel.append(nparray1.tolist())
                                            cc += 1
                                            if frame >= 2:
                                                # Particle acceleration (excluding rigid particles)
                                                nparray2 = np.subtract(
                                                    np.array(particle_vel[frame-1]),
                                                    np.array(particle_vel[frame-2])) #* frame_rate
                                                particle_acc.append(nparray2.tolist())
                                            # Visualize velocity distribution
                                            if VISUALIZATION:
                                                xvel = []
                                                yvel = []
                                                zvel = []
                                                xacc = []
                                                yacc = []
                                                zacc = []
                                                if cc % 60 == 0:
                                                    for i in range(len(particle_vel[-1])):
                                                        xvel.append(particle_vel[-1][i][0])
                                                        yvel.append(particle_vel[-1][i][1])
                                                        zvel.append(particle_vel[-1][i][2])
                                                        if i < len(particle_acc[-1]):
                                                            xacc.append(particle_acc[-1][i][0])
                                                            yacc.append(particle_acc[-1][i][1])
                                                            zacc.append(particle_acc[-1][i][2])
                                                    plt.figure(1)
                                                    fig, axs = plt.subplots(3)
                                                    fig.suptitle('Velocity histogram')
                                                    axs[0].hist(xvel, bins=100, label='Vx')
                                                    axs[0].legend("x", loc="upper right")
                                                    axs[1].hist(yvel, bins=100, label='Vy')
                                                    axs[1].legend("y", loc="upper right")
                                                    axs[2].hist(zvel, bins=100, label='Vz')
                                                    axs[2].legend("z", loc="upper right")
                                                    plt.savefig(ds_path_save+'/vel_'+str(c1)+'_'+str(cc)+'.png', dpi=1000)
                                                    plt.figure(2)
                                                    fig, axs = plt.subplots(3)
                                                    fig.suptitle('Acceleration histogram')
                                                    axs[0].hist(xacc, bins=100, label='Vx')
                                                    axs[0].legend("x", loc="upper right")
                                                    axs[1].hist(yacc, bins=100, label='Vy')
                                                    axs[1].legend("y", loc="upper right")
                                                    axs[2].hist(zacc, bins=100, label='Vz')
                                                    axs[2].legend("z", loc="upper right")
                                                    plt.savefig(ds_path_save+'/acc_'+str(c1)+'_'+str(cc)+'.png', dpi=1000)

                                        # Calculate mean and variance
                                        xvel = []
                                        yvel = []
                                        zvel = []
                                        xacc = []
                                        yacc = []
                                        zacc = []
                                        for i in range(len(particle_vel)):
                                            for j in range(len(particle_vel[-1])):
                                                xvel.append(particle_vel[i][j][0])
                                                yvel.append(particle_vel[i][j][1])
                                                zvel.append(particle_vel[i][j][2])
                                                if i < len(particle_acc) and j < len(particle_acc[-1]):
                                                    xacc.append(particle_acc[i][j][0])
                                                    yacc.append(particle_acc[i][j][1])
                                                    zacc.append(particle_acc[i][j][2])
                                        # Mean
                                        particle_vel_mean_case = ([np.mean(xvel), np.mean(yvel), np.mean(zvel)])
                                        particle_acc_mean_case = ([np.mean(xacc), np.mean(yacc), np.mean(zacc)])
                                        rigid_body_force_mean_case = (np.mean(rigid_body_force, axis=0).tolist())
                                        particle_vel_mean_old = particle_vel_mean
                                        particle_acc_mean_old = particle_acc_mean
                                        rigid_body_force_mean_old = rigid_body_force_mean
                                        # Variance (Ref: https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html)
                                        particle_vel_var_case = ([np.var(xvel), np.var(yvel), np.var(zvel)])
                                        particle_acc_var_case = ([np.var(xacc), np.var(yacc), np.var(zacc)])
                                        rigid_body_force_var_case = (np.var(rigid_body_force, axis=0).tolist())
                                        particle_vel_var_old = particle_vel_var
                                        particle_acc_var_old = particle_acc_var
                                        rigid_body_force_var_old = rigid_body_force_var
                                        n1_v += n2_v
                                        n2_v = len(xvel)
                                        n1_a += n2_a
                                        n2_a = len(xacc)
                                        n1_f += n2_f
                                        n2_f = len(rigid_body_force)
                                        if particle_vel_mean[0] == 0:
                                            particle_vel_mean = particle_vel_mean_case
                                            particle_acc_mean = particle_acc_mean_case
                                            rigid_body_force_mean = rigid_body_force_mean_case
                                            particle_vel_var = particle_vel_var_case
                                            particle_acc_var = particle_acc_var_case
                                            rigid_body_force_var = rigid_body_force_var_case
                                        else:
                                            particle_vel_mean = np.divide(
                                                np.add(np.multiply(particle_vel_mean_old, n1_v), np.multiply(particle_vel_mean_case, n2_v)), n1_v + n2_v)
                                            particle_acc_mean = np.divide(
                                                np.add(np.multiply(particle_acc_mean_old, n1_a), np.multiply(particle_acc_mean_case, n2_a)), n1_a + n2_a)
                                            rigid_body_force_mean = np.divide(
                                                np.add(np.multiply(rigid_body_force_mean_old, n1_f), np.multiply(rigid_body_force_mean_case, n2_f)), n1_f + n2_f)

                                            if (n1_v + n2_v) > 0:
                                                particle_vel_var = np.divide(np.add(
                                                    np.multiply(n1_v, np.add(particle_vel_var_old, np.power(np.subtract(particle_vel_mean_old, particle_vel_mean), 2))),
                                                    np.multiply(n2_v, np.add(particle_vel_var_case, np.power(np.subtract(particle_vel_mean_case, particle_vel_mean), 2)))),
                                                    n1_v + n2_v)
                                            if (n1_a + n2_a) > 0:
                                                particle_acc_var = np.divide(np.add(
                                                    np.multiply(n1_a, np.add(particle_acc_var_old, np.power(np.subtract(particle_acc_mean_old, particle_acc_mean), 2))),
                                                    np.multiply(n2_a, np.add(particle_acc_var_case, np.power(np.subtract(particle_acc_mean_case, particle_acc_mean), 2)))),
                                                    n1_a + n2_a)
                                            rigid_body_force_var = np.divide(np.add(
                                                np.multiply(n1_f, np.add(rigid_body_force_var_old, np.power(np.subtract(rigid_body_force_mean_old, rigid_body_force_mean), 2))),
                                                np.multiply(n2_f, np.add(rigid_body_force_var_case, np.power(np.subtract(rigid_body_force_mean_case, rigid_body_force_mean), 2)))),
                                                n1_f + n2_f)

                                    c2 += 1
                                    c3 += 1

                                if c2 >= len_split:
                                    break
                            if c2 >= len_split:
                                break
                        if c2 >= len_split:
                            break
                    if c2 >= len_split:
                        break


        ''' Create metadata json file '''
        if split == 'train':
            print("Vel mean:", np.array(particle_vel_mean).tolist())
            print("Vel std:", np.sqrt(particle_vel_var).tolist())
            print("Acc mean:", np.array(particle_acc_mean).tolist())
            print("Acc std:", np.sqrt(particle_acc_var).tolist())
            print("Force mean:", np.array(rigid_body_force_mean).tolist())
            print("Force std:", np.sqrt(rigid_body_force_var).tolist())
            
            rigid_body_ref_point = int(r_particles/2)

            tfrecord_excav.write_json(
                dataset_save,
                bounds, frames, dimension, frame_rate,
                np.array(particle_vel_mean).tolist(), np.sqrt(particle_vel_var).tolist(),
                np.array(particle_acc_mean).tolist(), np.sqrt(particle_acc_var).tolist(),
                np.array(rigid_body_force_mean).tolist(), np.sqrt(rigid_body_force_var).tolist(),
                dataset_split,
                rigid_body_ref_point, r_particles, MODE_NUMBER)
