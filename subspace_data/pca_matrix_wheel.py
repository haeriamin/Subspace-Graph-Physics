'''
Example usage (from parent directory):
`python -m subspace_data.pca_matrix_wheel`

'''

import timeit
import os
import numpy as np
import pickle

from subspace_data import tfrecord_excav
from subspace_data import pca


if __name__ == '__main__':

    # If dimensional analysis used
    scale_dim = 0.25
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2

    # Simulation properties
    dataset_load = 'Wheel'
    dataset_save = 'Wheel_PCA'

    dataset_split = 'train'  # 'train', 'valid' or 'test'
    frame_rate = 120 * scale_time
    dimension = 3

    dataset_load = './learning_to_simulate/datasets/' + dataset_load
    dataset_save = './learning_to_simulate/datasets/' + dataset_save
    if not os.path.exists(dataset_save):
        os.makedirs(dataset_save)

    case_number = 0
    particles = 0
    frames = 320

    c = 0
    for gravity in [1.62, 3.72, 9.81]:
        for slip in [20, 40, 70]:
            for wload in [100, 164, 225]:
                for wdia in [0.05, 0.15, 0.30]:
                    for sfangle in [30, 37, 43]:
                        c += 1
                        if c == 240:
                            if wdia == 0.05:
                                load = wload / 9
                            else:
                                load = wload
                            task_id = str(c)+'_G'+str(gravity)+'ms-2_S'+str(int(slip)
                                )+'perc_L'+str(int(load))+'N_D'+str(int(wdia*100))+'cm_A'+str(int(sfangle))+'deg'
                            ds_path = os.path.join(dataset_load, task_id)
                            path, dirs, files = next(os.walk(os.path.join(ds_path, 'frames')))
                            file_path = os.path.join(
                                ds_path, 'frames/ds_' + "{0:0=4d}".format(1) + '.csv')
                            data_frame = tfrecord_excav.read_csv(file_path)
                            all_particles = len(data_frame)
                            for particle in range(1, all_particles):
                                if (int(data_frame['col0'][particle]) == 0):# and (particle%2 == 0):
                                    particles += 1
                        # if c == 240 or c == 118:  # for test
                        if c > 0:
                            case_number += 1

    X = np.zeros((case_number*(frames+1)*dimension, particles))
    print(len(X))
    print(len(X[0]))

    c = 0
    cc = 0
    for gravity in [1.62, 3.72, 9.81]:
        for slip in [20, 40, 70]:
            for wload in [100, 164, 225]:
                for wdia in [0.05, 0.15, 0.30]:
                    for sfangle in [30, 37, 43]:
                        c += 1
                        # if c == 240 or c == 118:  # for test
                        if c > 0:
                            cc += 1
                            if wdia == 0.05:
                                load = wload / 9
                            else:
                                load = wload
                            task_id = str(c)+'_G'+str(gravity)+'ms-2_S'+str(int(slip)
                                )+'perc_L'+str(int(load))+'N_D'+str(int(wdia*100))+'cm_A'+str(int(sfangle))+'deg'
                            print(task_id)
                            ds_path = os.path.join(dataset_load, task_id)
                            path, dirs, files = next(os.walk(os.path.join(ds_path, 'frames')))
                            
                            particle_pos = []
                            for frame in range(241, 240+frames+1):
                                file_path = os.path.join(
                                    ds_path, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')
                                data_frame = tfrecord_excav.read_csv(file_path)
                                all_particles = len(data_frame)
                                particle_pos_frame = []

                                for particle in range(1, all_particles):
                                    if ~np.isnan(float(data_frame['col1'][particle])):
                                        if (int(data_frame['col0'][particle]) == 0):# and (particle%2 == 0):
                                            particle_pos_frame.append([
                                                round(float(data_frame['col1'][particle])/scale_dim, 6),
                                                round(float(data_frame['col2'][particle])/scale_dim, 6),
                                                round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                    else:
                                        particle_pos_frame.append(particle_pos[frame-1][particle])

                                particle_pos.append(particle_pos_frame.copy())

                                if frame == 241:  # Extra frame in the beginning
                                    particle_pos.append(particle_pos_frame.copy())

                                if (-240+frame) % 10 == 0:
                                    print(c, ':', round((-240+frame)/frames, 3)*100, '%')

                            del particle_pos_frame
                            del data_frame

                            # Convert 3D data to 2D data
                            for i in range((frames+1)):
                                for j in range(particles):
                                    for k in range(dimension):
                                        X[k+i*dimension + (cc-1)*(frames+1)*dimension][j] = particle_pos[i][j][k]

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
