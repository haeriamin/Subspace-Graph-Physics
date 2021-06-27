'''
Example usage (from parent directory):
`python -m subspace_data.run_tfrecord_wheel_pca`

'''

# import pickle
import os
import struct
import pickle
import random
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from subspace_data import tfrecord_wheel
from subspace_data import pca_plot, plot
from subspace_data import pca


MODE_NUMBER = 8
VISUALIZATION = False


if __name__ == '__main__':
    dataset_load = './learning_to_simulate/datasets/' + 'Wheel'
    dataset_save = dataset_load + '_PCA'


    # Shuffle and create train, valid, and test examples (cases)
    no_examples = 3*3*3*3*3  # 243
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

    # train_example_list = [240,12]
    dataset_split = {
        'train': train_example_list,
        'valid': valid_example_list,
        'test': test_example_list
    }
    print(dataset_split)


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
    wwidth = 0.125  # wheel width [m]
    

    # Start
    for counter, split in enumerate(dataset_split):
        # Initialization 
        particle_key = 0
        c2, c3, frames = 0, 0, 0
        if split == 'train':   
            particle_vel_mean, particle_vel_var = [0., 0., 0.], [0., 0., 0.]
            particle_acc_mean, particle_acc_var = [0., 0., 0.], [0., 0., 0.]
            rigid_body_force_mean, rigid_body_force_var = [0., 0., 0.], [0., 0., 0.]
            step_context_mean, step_context_var = [0., 0., 0.], [0., 0., 0.]
            n1_v, n2_v = 0, 0
            n1_a, n2_a = 0, 0
            n1_f, n2_f = 0, 0
            n1_g, n2_g = 0, 0

        tfr_filename = os.path.join(dataset_save, f'{split}.tfrecord')
        with tf.python_io.TFRecordWriter(tfr_filename) as writer:
            
            len_split = len(dataset_split[split])
            while c2 < len_split:

                c1 = 0
                for gravity in [1.62, 3.72, 9.81]:
                    for slip in [20, 40, 70]:
                        for wload in [100, 164, 225]:
                            for wdia in [0.05, 0.15, 0.30]:
                                for sfangle in [30, 37, 43]:
                                    c1 += 1

                                    if dataset_split[split][c3] == c1:

                                        if wdia == 0.15:
                                            width = wwidth * 2/3
                                            load = wload
                                        elif wdia == 0.05:
                                            width = wwidth * 2/3
                                            load = wload / 9
                                        else:
                                            width = wwidth
                                            load = wload
                                        task_id = str(c1)+'_G'+str(gravity)+'ms-2_S'+str(int(slip)
                                            )+'perc_L'+str(int(load))+'N_D'+str(int(wdia*100))+'cm_A'+str(int(sfangle))+'deg'

                                        ds_path_load = os.path.join(dataset_load, task_id)
                                        if VISUALIZATION:
                                            ds_path_save = os.path.join(dataset_save, task_id)
                                            if not os.path.exists(ds_path_save):
                                                os.makedirs(ds_path_save)

                                        path, dirs, files = next(os.walk(os.path.join(ds_path_load, 'frames')))
                                        frames = min(320, len(files)-241)

                                        particle_id = []
                                        particle_type = []
                                        particle_pos = []
                                        rigid_body_force = []
                                        step_context = []
                                        side_wheel_particle = None
                                        if VISUALIZATION:
                                            particle_type_np = []
                                            particle_pos_np = []
                                        print(task_id)
                                        for frame in range(241, 240+frames+1):
                                            r_particles = 0
                                            particle_id_frame = []
                                            particle_type_frame = []
                                            particle_pos_frame = []
                                            r_particle_pos_x_min = 10
                                            r_particle_pos_y_min = 10
                                            r_particle_pos_z_min = 10
                                            if VISUALIZATION:
                                                particle_type_frame_np = []
                                                particle_pos_frame_np = []

                                            file_path = os.path.join(
                                                ds_path_load, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')

                                            data_frame = tfrecord_wheel.read_csv(file_path)
                                            all_particles = len(data_frame)


                                            ''' Create structured rigid body '''
                                            # Get center of wheel
                                            meanx, meany, meanz = 0, 0, 0
                                            counter = 0
                                            for particle in reversed(range(1, all_particles)):
                                                if int(data_frame['col0'][particle]) == 1:
                                                    meanx += round(float(data_frame['col1'][particle])/scale_dim, 6)
                                                    meany += round(float(data_frame['col2'][particle])/scale_dim, 6)
                                                    meanz += round(float(data_frame['col3'][particle])/scale_dim, 6)
                                                    counter += 1
                                            meanx /= counter
                                            meany /= counter
                                            meanz /= counter
                                            # Get one particle on side of wheel => side_wheel_particle = 6305
                                            if frame == 241:
                                                r_particle_pos_z_min = 10
                                                for particle in reversed(range(1, all_particles)):
                                                    if int(data_frame['col0'][particle]) == 1:
                                                        minz = round(float(data_frame['col3'][particle])/scale_dim, 6)
                                                        if minz < r_particle_pos_z_min:
                                                            r_particle_pos_z_min = minz
                                                            side_wheel_particle = particle
                                                # print(r_particle_pos_z_min, side_wheel_particle)
                                            # Create structured wheel                           
                                            def rotate_point(cx, cy, angle, p):
                                                s = np.sin(np.deg2rad(angle))
                                                c = np.cos(np.deg2rad(angle))
                                                # translate point back to origin:
                                                px = p[0] - cx
                                                py = p[1] - cy
                                                # rotate point
                                                xnew = px * c - py * s
                                                ynew = px * s + py * c
                                                # translate point back:
                                                px = xnew + cx
                                                py = ynew + cy
                                                return [px, py, p[2]]
                                            rpos = []
                                            rpos.append([
                                                round(float(data_frame['col1'][side_wheel_particle])/scale_dim, 6),
                                                round(float(data_frame['col2'][side_wheel_particle])/scale_dim, 6),
                                                round(float(data_frame['col3'][side_wheel_particle])/scale_dim, 6)])
                                            rpos.append(rotate_point(meanx, meany, 45, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 90, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 135, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 180, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 225, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 270, rpos[0]))
                                            rpos.append(rotate_point(meanx, meany, 315, rpos[0]))
                                            # rpos.append([rpos[0][0], rpos[0][1], rpos[0][2]+width/2])
                                            # rpos.append([rpos[1][0], rpos[1][1], rpos[1][2]+width/2])
                                            # rpos.append([rpos[2][0], rpos[2][1], rpos[2][2]+width/2])
                                            # rpos.append([rpos[3][0], rpos[3][1], rpos[3][2]+width/2])
                                            # rpos.append([rpos[4][0], rpos[4][1], rpos[4][2]+width/2])
                                            # rpos.append([rpos[5][0], rpos[5][1], rpos[5][2]+width/2])
                                            # rpos.append([rpos[6][0], rpos[6][1], rpos[6][2]+width/2])
                                            # rpos.append([rpos[7][0], rpos[7][1], rpos[7][2]+width/2])
                                            rpos.append([rpos[0][0], rpos[0][1], rpos[0][2]+width])
                                            rpos.append([rpos[1][0], rpos[1][1], rpos[1][2]+width])
                                            rpos.append([rpos[2][0], rpos[2][1], rpos[2][2]+width])
                                            rpos.append([rpos[3][0], rpos[3][1], rpos[3][2]+width])
                                            rpos.append([rpos[4][0], rpos[4][1], rpos[4][2]+width])
                                            rpos.append([rpos[5][0], rpos[5][1], rpos[5][2]+width])
                                            rpos.append([rpos[6][0], rpos[6][1], rpos[6][2]+width])
                                            rpos.append([rpos[7][0], rpos[7][1], rpos[7][2]+width])
                                            for p in rpos:
                                                r_particles += 1
                                                particle_id_frame.append(-1)
                                                particle_pos_frame.append(p)
                                                particle_type_frame.append(0)  # Particle type: boundary: 3, rigid: 0
                                                if VISUALIZATION:
                                                    particle_pos_frame_np.append(np.array(p))
                                                    particle_type_frame_np.append(np.array(0))


                                            " Add soil particles "
                                            for particle in range(1, all_particles):
                                                if ~np.isnan(float(data_frame['col1'][particle])):
                                                    if (int(data_frame['col0'][particle]) == 0):# and (particle%2 == 0):
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
                                            step_context.append([
                                                round(gravity, 6),
                                                round(sfangle, 6),
                                                round(load, 6)
                                            ])
                                            if VISUALIZATION:
                                                particle_type_np.append(np.array(particle_type_frame_np))
                                                particle_pos_np.append(np.array(particle_pos_frame_np))
                                            # Extra frame in the beginning
                                            if frame == 241:
                                                particle_id.append(particle_id_frame.copy())
                                                particle_type.append(particle_type_frame.copy())
                                                particle_pos.append(particle_pos_frame.copy())
                                                rigid_body_force.append([
                                                    round(float(data_frame['col0'][0])/(scale_dim**3), 6),
                                                    round(float(data_frame['col1'][0])/(scale_dim**3), 6),
                                                    round(float(data_frame['col2'][0])/(scale_dim**3), 6)])
                                                step_context.append([
                                                    round(gravity, 6),
                                                    round(sfangle, 6),
                                                    round(load, 6)
                                                ])
                                                if VISUALIZATION:
                                                    particle_type_np.append(np.array(particle_type_frame_np))
                                                    particle_pos_np.append(np.array(particle_pos_frame_np))
                                            if (frame-240) % 10 == 0:
                                                print(c2+1, ':', round((frame-240)/frames, 3)*100, '%')
                                        del particle_pos_frame
                                        del data_frame
                                        particles = len(particle_pos[0])

                                        # print(r_particles)
                                        # print(particles)
                                        if VISUALIZATION:
                                            pca_plot.plot(particle_pos_np, particle_type_np)


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
                                        step_context_bytes = []

                                        particle_type_bytes = struct.pack(
                                            '%sq' % len(particle_type[0][:r_particles+MODE_NUMBER]), *particle_type[0][:r_particles+MODE_NUMBER])

                                        for i in range(frames+1):
                                            particle_pos_bytes.append(struct.pack(
                                                '%sf' % len(particle_pos_2D[i]), *particle_pos_2D[i]))

                                            rigid_body_force_bytes.append(struct.pack(
                                                '%sf' % len(rigid_body_force[i]), *rigid_body_force[i]))

                                            step_context_bytes.append(struct.pack(
                                                '%sf' % len(step_context[i]), *step_context[i]))

                                        data = {
                                            'context1': particle_type_bytes,
                                            'context2': particle_key,
                                            'feature1': particle_pos_bytes,
                                            'feature2': rigid_body_force_bytes,
                                            'feature3': step_context_bytes,
                                        }
                                        del particle_type
                                        del particle_pos_2D
                                        del particle_type_bytes
                                        del particle_pos_bytes
                                        del rigid_body_force_bytes
                                        del particle_pos

                                        example = tfrecord_wheel.serialize(data)
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
                                            step_context_mean_case = (np.mean(step_context, axis=0).tolist())
                                            particle_vel_mean_old = particle_vel_mean
                                            particle_acc_mean_old = particle_acc_mean
                                            rigid_body_force_mean_old = rigid_body_force_mean
                                            step_context_mean_old = step_context_mean
                                            # Combined variance (Ref: https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html)
                                            particle_vel_var_case = ([np.var(xvel), np.var(yvel), np.var(zvel)])
                                            particle_acc_var_case = ([np.var(xacc), np.var(yacc), np.var(zacc)])
                                            rigid_body_force_var_case = (np.var(rigid_body_force, axis=0).tolist())
                                            step_context_var_case = (np.var(step_context, axis=0).tolist())
                                            particle_vel_var_old = particle_vel_var
                                            particle_acc_var_old = particle_acc_var
                                            rigid_body_force_var_old = rigid_body_force_var
                                            step_context_var_old = step_context_var
                                            n1_v += n2_v
                                            n2_v = len(xvel)
                                            n1_a += n2_a
                                            n2_a = len(xacc)
                                            n1_f += n2_f
                                            n2_f = len(rigid_body_force)
                                            n1_g += n2_g
                                            n2_g = len(step_context)
                                            if particle_vel_mean[0] == 0:
                                                particle_vel_mean = particle_vel_mean_case
                                                particle_acc_mean = particle_acc_mean_case
                                                rigid_body_force_mean = rigid_body_force_mean_case
                                                step_context_mean = step_context_mean_case
                                                particle_vel_var = particle_vel_var_case
                                                particle_acc_var = particle_acc_var_case
                                                rigid_body_force_var = rigid_body_force_var_case
                                                step_context_var = step_context_var_case
                                            else:
                                                particle_vel_mean = np.divide(
                                                    np.add(np.multiply(particle_vel_mean_old, n1_v), np.multiply(particle_vel_mean_case, n2_v)), n1_v + n2_v)
                                                particle_acc_mean = np.divide(
                                                    np.add(np.multiply(particle_acc_mean_old, n1_a), np.multiply(particle_acc_mean_case, n2_a)), n1_a + n2_a)
                                                rigid_body_force_mean = np.divide(
                                                    np.add(np.multiply(rigid_body_force_mean_old, n1_f), np.multiply(rigid_body_force_mean_case, n2_f)), n1_f + n2_f)
                                                step_context_mean = np.divide(
                                                    np.add(np.multiply(step_context_mean_old, n1_g), np.multiply(step_context_mean_case, n2_g)), n1_g + n2_g)
                                                
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

                                                step_context_var = np.divide(np.add(
                                                    np.multiply(n1_g, np.add(step_context_var_old, np.power(np.subtract(step_context_mean_old, step_context_mean), 2))),
                                                    np.multiply(n2_g, np.add(step_context_var_case, np.power(np.subtract(step_context_mean_case, step_context_mean), 2)))),
                                                    n1_g + n2_g)
                                        
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
            print("Context mean:", np.array(step_context_mean).tolist())
            print("Context std:", np.sqrt(step_context_var).tolist())
            
            rigid_body_ref_point = None

            tfrecord_wheel.write_json(
                dataset_save,
                bounds, frames, dimension, frame_rate,
                np.array(particle_vel_mean).tolist(), np.sqrt(particle_vel_var).tolist(),
                np.array(particle_acc_mean).tolist(), np.sqrt(particle_acc_var).tolist(),
                np.array(rigid_body_force_mean).tolist(), np.sqrt(rigid_body_force_var).tolist(),
                dataset_split,
                np.array(step_context_mean).tolist(), np.sqrt(step_context_var).tolist(),
                rigid_body_ref_point, r_particles, MODE_NUMBER)
