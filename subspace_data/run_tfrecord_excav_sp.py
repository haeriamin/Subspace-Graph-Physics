'''
Example usage (from parent directory):
`python -m subspace_data.run_tfrecord_excav_sp`

Ref:
https://docs.python.org/3/library/struct.html

'''

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
# import csv

from subspace_data import tfrecord_excav
from subspace_data import plot


if __name__ == '__main__':

    VISUALIZATION = False

    # If dimensional analysis used
    scale_dim = 0.25
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2

    # Simulation properties
    dataset_load = 'Excavation'
    dataset_save = 'Excavation_SP'

    dataset_split = 'train'  # 'train', 'valid' or 'test'
    frame_rate = 120 * scale_time
    dimension = 3
    bounds = [[0.1, 0.9],
              [0.2, 1.0],
              [0.1, 0.9]]
    soil_height = (0.025 + 0.0375)/scale_dim
    plate_height = 0.0763/scale_dim
    plate_width = 0.1142/scale_dim

    c = 0
    particle_key = 0
    dataset_load = './learning_to_simulate/datasets/' + dataset_load
    dataset_save = './learning_to_simulate/datasets/' + dataset_save

    particle_vel_mean = [0., 0., 0.]
    particle_vel_var = [0., 0., 0.]
    particle_acc_mean = [0., 0., 0.]
    particle_acc_var = [0., 0., 0.]
    rigid_body_force_mean = [0., 0., 0.]
    rigid_body_force_var = [0., 0., 0.]
    n1_v = 0
    n2_v = 0
    n1_a = 0
    n2_a = 0
    n1_f = 0
    n2_f = 0

    frames = 0

    for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
        for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
            for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                for motion in [1.0, 2.0]:
                    c += 1
                    # if c == 241 or c == 242:
                    if speed != 0.01:  # exclude 50 bad cases

                        task_id = str(c) + '_D' + str(depth) + 'm_S' + str(
                            speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)

                        ds_path_load = os.path.join(dataset_load, task_id)
                        
                        ds_path_save = os.path.join(dataset_save, task_id)
                        if not os.path.exists(ds_path_save):
                            os.makedirs(ds_path_save)

                        path, dirs, files = next(os.walk(os.path.join(ds_path_load, 'frames')))
                        frames = min(320, len(files)-1)

                        particles = []
                        particle_id = []
                        particle_type = []
                        particle_pos = []
                        rigid_body_force = []
                        added_particle_pos = []
                        if VISUALIZATION:
                            particle_type_np = []
                            particle_pos_np = []
                        print(task_id)
                        particle_id_frame = []
                        for frame in range(1, frames+1):
                            r_particles = 0
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
                            grid_size = int(res*0.02)  # [m]
                            for iz in range(int(r_particle_pos_z_min*res), int((r_particle_pos_z_min+plate_width)*res), grid_size):
                                for iy in range(int(r_particle_pos_y_min*res), int((r_particle_pos_y_min + plate_height*np.cos(np.deg2rad(angle)))*res), grid_size):
                                    r_particles += 1
                                    # particle_id_frame.append(-1)
                                    particle_pos_frame.append([
                                        # 1.0 - round(data_frame['col1'][particle]/scale_dim, 6),
                                        r_particle_pos_x_min + np.tan(np.deg2rad(angle))*(iy/res-r_particle_pos_y_min),
                                        iy/res,
                                        iz/res])
                                    # Particle type: soil: 6, boundary: 3, rigid: 0
                                    particle_type_frame.append(3)
                                    if VISUALIZATION:
                                        particle_pos_frame_np.append(np.array([
                                            # 1.0 - round(data_frame['col1'][particle]/scale_dim, 6),
                                            r_particle_pos_x_min + np.tan(np.deg2rad(angle))*(iy/res-r_particle_pos_y_min),
                                            iy/res,
                                            iz/res]))
                                        particle_type_frame_np.append(np.array(3))

                            if frame > 1:
                                for particle in particle_id_frame:
                                    if ~np.isnan(float(data_frame['col1'][particle])):
                                        particle_pos_frame.append([
                                            round(float(data_frame['col1'][particle])/scale_dim, 6),
                                            round(float(data_frame['col2'][particle])/scale_dim, 6),
                                            round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                        # Particle type: soil: 6, boundary: 3, rigid: 0
                                        particle_type_frame.append(6)
                                        if VISUALIZATION:
                                            particle_pos_frame_np.append(np.array([
                                                round(float(data_frame['col1'][particle])/scale_dim, 6),
                                                round(float(data_frame['col2'][particle])/scale_dim, 6),
                                                round(float(data_frame['col3'][particle])/scale_dim, 6)]))
                                            particle_type_frame_np.append(np.array(6))
                                    else:
                                        particle_type_frame_np.append(np.array(6))
                                        particle_pos_frame.append(particle_pos[frame-1][particle])

                            for particle in range(1, all_particles):
                                if ~np.isnan(float(data_frame['col1'][particle])):
                                    if (round(float(data_frame['col2'][particle])/scale_dim, 6) > soil_height) and (int(data_frame['col0'][particle]) == 0) and (particle%5 == 0) and (particle not in particle_id_frame):
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
                                    if particle not in particle_id_frame:
                                        particle_id_frame.append(particle)
                                        particle_type_frame_np.append(np.array(6))
                                        particle_pos_frame.append(particle_pos[frame-1][particle])

                            particles.append(len(particle_pos_frame))
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
                            # Extra frame in the beginning
                            if frame == 1:
                                particles.append(len(particle_pos_frame))
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
                                print(c, ':', round(frame/frames, 3)*100, '%')

                        del particle_pos_frame
                        del data_frame

                        # with open('id.csv', 'w') as f:
                        #     write = csv.writer(f)
                        #     # for i in range(len(particle_pos_np)):
                        #     #     a = np.reshape(particle_pos_np[i], (3*particles[i]))
                        #     #     write.writerow(a[r_particles:r_particles+1000])
                        #     #     print(a[r_particles:])
                        #     #     print("")
                        #     for i in range(len(particle_id)):
                        #         write.writerow(particle_id[i])
                        #         # print(particle_id[i])
                        #         # print("")

                        # Pad particle positions
                        max_particles = max(particles)
                        print("# of particles:", max_particles)
                        for frame in range(0, frames):
                            for id in particle_id[frame+1][len(particle_id[frame]):]:
                                # if id not in particle_id[frame]:
                                index = particle_id[frame+1].index(id)
                                new_particle_pos = [
                                    particle_pos[frame+1][r_particles+index][0],
                                    particle_pos[frame+1][r_particles+index][1], #- 0.005
                                    particle_pos[frame+1][r_particles+index][2]]
                                added_particle_pos.append(new_particle_pos.copy())
                                for fr in range(frame+1):
                                    particle_id[fr].append(id)
                                    particle_pos[fr].append(new_particle_pos.copy())
                                    particle_type[fr].append(6)
                                    if VISUALIZATION:
                                        particle_pos_np[fr] = np.append(particle_pos_np[fr], np.array([new_particle_pos]), axis=0)
                                        particle_type_np[fr] = np.append(particle_type_np[fr], np.array([10]), axis=0)

                        # # Calculate minimum particle distance to each particle => ~2cm => R=0.03m is the best
                        # for i in range(len(particle_pos[0])):
                        #     dist_i = 100
                        #     for j in range(len(particle_pos[0])):
                        #         if i != j:
                        #             dist_j = np.linalg.norm(np.array(particle_pos[0][j])-np.array(particle_pos[0][i]))
                        #             if dist_j < dist_i:
                        #                 dist_i = dist_j
                        #     print(dist_i*100)

                        # Visualize inactive particle position distribution
                        xpos = []
                        ypos = []
                        zpos = []
                        for i in range(len(added_particle_pos)):
                            xpos.append(added_particle_pos[i][0])
                            ypos.append(added_particle_pos[i][1])
                            zpos.append(added_particle_pos[i][2])
                        plt.figure(0)
                        fig, axs = plt.subplots(3)
                        fig.suptitle('Inactive particle positions histogram')
                        axs[0].hist(xpos, bins=100, label='Vx')
                        axs[0].legend("x", loc="upper right")
                        axs[1].hist(ypos, bins=100, label='Vy')
                        axs[1].legend("y", loc="upper right")
                        axs[2].hist(zpos, bins=100, label='Vz')
                        axs[2].legend("z", loc="upper right")
                        plt.savefig(ds_path_save+'/inactive_pos.png', dpi=300)

                        if VISUALIZATION:
                            plot.plot(particle_pos_np, particle_type_np)
                            del particle_type_frame_np
                            del particle_pos_frame_np
                            del particle_pos_np
                            del particle_type_np

                        # Calculate particle velocities and accelerations
                        cc = 1
                        particle_vel = []
                        particle_acc = []
                        for frame in range(1, frames):
                            # Particle velocity (excluding rigid particles)
                            nparray1 = np.subtract(
                                np.array(particle_pos[frame][r_particles:]),
                                np.array(particle_pos[frame-1][r_particles:])) #* frame_rate
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
                                    plt.savefig(ds_path_save+'/vel_'+str(c)+'_'+str(cc)+'.png', dpi=1000)
                                    plt.figure(2)
                                    fig, axs = plt.subplots(3)
                                    fig.suptitle('Acceleration histogram')
                                    axs[0].hist(xacc, bins=100, label='Vx')
                                    axs[0].legend("x", loc="upper right")
                                    axs[1].hist(yacc, bins=100, label='Vy')
                                    axs[1].legend("y", loc="upper right")
                                    axs[2].hist(zacc, bins=100, label='Vz')
                                    axs[2].legend("z", loc="upper right")
                                    plt.savefig(ds_path_save+'/acc_'+str(c)+'_'+str(cc)+'.png', dpi=1000)

                        # with open('id.csv', 'w') as f:
                        #     write = csv.writer(f)
                        #     # for i in range(len(particle_pos_np)):
                        #     #     a = np.reshape(particle_pos_np[i], (3*particles[i]))
                        #     #     write.writerow(a[r_particles:r_particles+1000])
                        #     #     print(a[r_particles:])
                        #     #     print("")
                        #     for i in range(len(particle_id)):
                        #         write.writerow(particle_id[i])
                        #         # print(particle_id[i])
                        #         # print("")

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


                        ''' Serialize via tfrecord '''
                        # Convert 3D data to 2D data
                        particle_pos_2D = np.zeros(
                                (frames+1, max_particles*dimension))
                        for i in range(frames+1):
                            for j in range(max_particles):
                                for k in range(dimension):
                                    particle_pos_2D[i][dimension*j+k] = particle_pos[i][j][k]

                        particle_type_bytes = []
                        particle_pos_bytes = []
                        rigid_body_force_bytes = []

                        ind = np.argmax(particles)
                        particle_type_bytes = struct.pack(
                                '%sq' % len(particle_type[ind]), *particle_type[ind])

                        for i in range(frames+1):
                            particle_pos_bytes.append(struct.pack(
                                '%sf' % len(particle_pos_2D[i]), *particle_pos_2D[i]))

                            rigid_body_force_bytes.append(struct.pack(
                                '%sf' % len(rigid_body_force[i]), *rigid_body_force[i]))

                        del particle_type
                        del particle_pos_2D
                        del rigid_body_force

                        data = {
                            'context1': particle_type_bytes,
                            'context2': particle_key,
                            'feature1': particle_pos_bytes,
                            'feature2': rigid_body_force_bytes,
                        }
                        del particle_type_bytes
                        del particle_pos_bytes
                        del rigid_body_force_bytes

                        tfrecord_excav.serialize(ds_path_save, dataset_split, data)

                        del particle_pos


    ''' Create metadata json file '''
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
        rigid_body_ref_point)
