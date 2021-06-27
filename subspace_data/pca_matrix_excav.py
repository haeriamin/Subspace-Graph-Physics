'''
Example usage (from parent directory):
`python -m subspace_data.pca_matrix_excav`

'''

import timeit
import os
import numpy as np
import pickle

from subspace_data import tfrecord_excav
from subspace_data import pca


if __name__ == '__main__':
    dataset_load = './learning_to_simulate/datasets/' + 'Excavation'
    dataset_save = dataset_load + '_PCA'
    if not os.path.exists(dataset_save):
        os.makedirs(dataset_save)

    # If dimensional analysis used
    scale_dim = 0.25
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2

    # Simulation properties
    frame_rate = 120 * scale_time
    dimension = 3
    soil_height = (0.025 + 0.0375)/scale_dim
    plate_height = 0.0763/scale_dim
    plate_width = 0.1142/scale_dim
        
    case_number = 0
    particles = 0
    frames = 320

    c = 0
    for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
        for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
            for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                for motion in [1.0, 2.0]:
                    c += 1
                    if c == 240:
                        task_id = str(c) + '_D' + str(depth) + 'm_S' + str(
                            speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)
                        ds_path = os.path.join(dataset_load, task_id)
                        path, dirs, files = next(os.walk(os.path.join(ds_path, 'frames')))
                        file_path = os.path.join(
                            ds_path, 'frames/ds_' + "{0:0=4d}".format(1) + '.csv')
                        data_frame = tfrecord_excav.read_csv(file_path)
                        all_particles = len(data_frame)
                        for particle in range(1, all_particles):
                            if (int(data_frame['col0'][particle]) == 0) and (particle%5 == 0):
                                particles += 1
                    # if c <= 12 and speed != 0.01:
                    # if c == 240 or c == 12:
                    if speed != 0.01:
                        case_number += 1

    X = np.zeros((case_number*(frames+1)*dimension, particles))
    print(len(X))
    print(len(X[0]))

    c = 0
    cc = 0
    for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
        for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
            for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                for motion in [1.0, 2.0]:
                    c += 1
                    # if c <= 12 and speed != 0.01:
                    # if c == 240 or c == 12:
                    if speed != 0.01:  # exclude 50 bad cases
                        cc += 1
                        task_id = str(c) + '_D' + str(depth) + 'm_S' + str(
                            speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)
                        ds_path = os.path.join(dataset_load, task_id)
                        path, dirs, files = next(os.walk(os.path.join(ds_path, 'frames')))
                        
                        particle_pos = []
                        print(task_id)
                        for frame in range(1, frames+1):
                            file_path = os.path.join(
                                ds_path, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')
                            data_frame = tfrecord_excav.read_csv(file_path)
                            particle_pos_frame = []

                            # # Get one particle in rigid body
                            # r_particles = 0
                            # r_particle_pos_x_min = 10
                            # r_particle_pos_y_min = 10
                            # r_particle_pos_z_min = 10
                            # for particle in reversed(range(1, particles)):
                            #     if int(data_frame['col0'][particle]) == 1:
                            #         minx = round(float(data_frame['col1'][particle])/scale_dim, 6)
                            #         miny = round(float(data_frame['col2'][particle])/scale_dim, 6)
                            #         minz = round(float(data_frame['col3'][particle])/scale_dim, 6)
                            #         if minx < r_particle_pos_x_min:
                            #             r_particle_pos_x_min = minx
                            #         if miny < r_particle_pos_y_min:
                            #             r_particle_pos_y_min = miny
                            #         if minz < r_particle_pos_z_min:
                            #             r_particle_pos_z_min = minz
                            # # Create structured rigid body
                            # res = 10000
                            # grid_size = int(res*0.02)  # [m]
                            # for iz in range(int(r_particle_pos_z_min*res), int((r_particle_pos_z_min+plate_width)*res), grid_size):
                            #     for iy in range(int(r_particle_pos_y_min*res), int((r_particle_pos_y_min + plate_height*np.cos(np.deg2rad(angle)))*res), grid_size):
                            #         r_particles += 1
                            #         particle_pos_frame.append([
                            #             r_particle_pos_x_min + np.tan(np.deg2rad(angle))*(iy/res-r_particle_pos_y_min),
                            #             iy/res,
                            #             iz/res])

                            for particle in range(1, all_particles):
                                if ~np.isnan(float(data_frame['col1'][particle])):
                                    if (int(data_frame['col0'][particle]) == 0) and (particle%5 == 0):
                                        particle_pos_frame.append([
                                            round(float(data_frame['col1'][particle])/scale_dim, 6),
                                            round(float(data_frame['col2'][particle])/scale_dim, 6),
                                            round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                else:
                                    particle_pos_frame.append(particle_pos[frame-1][particle])

                            particle_pos.append(particle_pos_frame.copy())

                            if frame == 1:  # Extra frame in the beginning
                                particle_pos.append(particle_pos_frame.copy())

                            if frame % 10 == 0:
                                print(c, ':', round(frame/frames, 3)*100, '%')

                        del particle_pos_frame
                        del data_frame

                        # Convert 3D data to 2D data
                        for i in range((frames+1)):
                            for j in range(particles):
                                for k in range(dimension):
                                    # X[i+(cc-1)*(frames+1)][k+j*dimension] = particle_pos[i][j][k]  # case 2 in pca.pdf
                                    X[k+i*dimension + (cc-1)*(frames+1)*dimension][j] = particle_pos[i][j][k]  # case 1 in pca.pdf


    ''' Process data '''
    # Apply PCA to 2D data
    start = timeit.default_timer()
    EigVec, EigVal, MEAN = pca.evd_np(X)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # Save PCA full transformation (loading) matrix, EigVec
    with open(dataset_save + '/' + 'pca_eigen_vectors.pkl', 'wb') as f:
        pickle.dump(EigVec, f)

    # Save PCA eigen values, EigVal
    with open(dataset_save + '/' + 'pca_eigen_values.pkl', 'wb') as f:
        pickle.dump(EigVal, f)

    # Save data mean scalar, MEAN
    with open(dataset_save + '/' + 'pca_mean_scalar.pkl', 'wb') as f:
        pickle.dump(MEAN, f)
