'''
Example usage (from parent directory):
`python -m subspace_data.error_computation_pca`

'''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from subspace_data import tfrecord_excav
from subspace_data import pca


if __name__ == '__main__':
    # If dimensional analysis used
    scale_dim = 0.25
    scale_time = scale_dim * 2
    scale_vel = scale_dim * 2


    ''' Excavation '''
    dataset_load = 'Excavation'
    dataset_save = 'Excavation_PCA'
    dataset_split = 'train'  # 'train', 'valid' or 'test'
    dimension = 3
    dataset_load = './learning_to_simulate/datasets/' + dataset_load
    dataset_save = './learning_to_simulate/datasets/' + dataset_save
    c = 0
    frames = 0
    for depth in [0.02, 0.04, 0.05, 0.08, 0.1]:
        for speed in [0.01, 0.04, 0.08, 0.1, 0.15]:
            for angle in [0.0, 3.8, 10.0, 30.8, 45.0]:
                for motion in [1.0, 2.0]:
                    c += 1
                    if c == 240:
                        task_id = str(c) + '_D' + str(depth) + 'm_S' + str(
                            speed) + 'ms-1_A' + str(angle) + 'deg_M' + str(motion)

                        ds_path_load = os.path.join(dataset_load, task_id)
                        ds_path_save = os.path.join(dataset_save, task_id)
                        if not os.path.exists(ds_path_save):
                            os.makedirs(ds_path_save)

                        path, dirs, files = next(os.walk(os.path.join(ds_path_load, 'frames')))
                        frames = min(320, len(files)-1)

                        particle_pos = []
                        print(task_id)
                        for frame in range(1, frames+1):
                            particle_pos_frame = []

                            file_path = os.path.join(
                                ds_path_load, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')

                            data_frame = tfrecord_excav.read_csv(file_path)
                            all_particles = len(data_frame)

                            for particle in range(1, all_particles):
                                if ~np.isnan(float(data_frame['col1'][particle])):
                                    if (int(data_frame['col0'][particle]) == 0) and (particle%5 == 0):
                                        particle_pos_frame.append([
                                            round(float(data_frame['col1'][particle])/scale_dim, 6),
                                            round(float(data_frame['col2'][particle])/scale_dim, 6),
                                            round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                else:
                                    particle_pos_frame.append(particle_pos[frame-1][particle])

                            particle_pos.append(particle_pos_frame)
                            # Extra frame in the beginning
                            if frame == 1:
                                particle_pos.append(particle_pos_frame.copy())
                            if frame % 10 == 0:
                                print(c, ':', round(frame/frames, 3)*100, '%')
                        particles = len(particle_pos[0])

    # Load PCA loading matrix and mean scalar
    with open(dataset_save + '/pca_eigen_vectors.pkl', 'rb') as f:
        W_excav = pickle.load(f)
    with open(dataset_save + '/pca_mean_scalar.pkl', 'rb') as f:
        MEAN_excav = pickle.load(f)
    with open(dataset_save + '/pca_eigen_values.pkl', 'rb') as f:
        V_excav = pickle.load(f)
    # Convert 3D soil data to 2D data
    X2D_excav = np.zeros((dimension*(frames+1), particles))
    for i in range((frames+1)):
        for j in range(particles):
            for k in range(dimension):
                X2D_excav[k+i*dimension][j] = particle_pos[i][j][k] - MEAN_excav


    ''' Wheel '''
    # Simulation properties
    dataset_load = 'Wheel'
    dataset_save = 'Wheel_PCA'
    dataset_split = 'train'  # 'train', 'valid' or 'test'
    dimension = 3
    dataset_load = './learning_to_simulate/datasets/' + dataset_load
    dataset_save = './learning_to_simulate/datasets/' + dataset_save
    wwidth = 0.125  # wheel width [m]
    c = 0
    frames = 0
    for gravity in [1.62, 3.72, 9.81]:
        for slip in [20, 40, 70]:
            for wload in [100, 164, 225]:
                for wdia in [0.05, 0.15, 0.30]:
                    for sfangle in [30, 37, 43]:
                        c += 1
                        if c == 240:
                            if wdia == 0.15:
                                width = wwidth * 2/3
                                load = wload
                            elif wdia == 0.05:
                                width = wwidth * 2/3
                                load = wload / 9
                            else:
                                width = wwidth
                                load = wload
                            task_id = str(c)+'_G'+str(gravity)+'ms-2_S'+str(int(slip)
                                )+'perc_L'+str(int(load))+'N_D'+str(int(wdia*100))+'cm_A'+str(int(sfangle))+'deg'

                            ds_path_load = os.path.join(dataset_load, task_id)
                            ds_path_save = os.path.join(dataset_save, task_id)
                            if not os.path.exists(ds_path_save):
                                os.makedirs(ds_path_save)

                            path, dirs, files = next(os.walk(os.path.join(ds_path_load, 'frames')))
                            frames = min(320, len(files)-241)

                            particle_pos = []
                            print(task_id)
                            for frame in range(241, 240+frames+1):
                                particle_pos_frame = []
                                file_path = os.path.join(
                                    ds_path_load, 'frames/ds_' + "{0:0=4d}".format(frame) + '.csv')
                                data_frame = tfrecord_excav.read_csv(file_path)
                                all_particles = len(data_frame)
                                " Add soil particles "
                                for particle in range(1, all_particles):
                                    if ~np.isnan(float(data_frame['col1'][particle])):
                                        if (int(data_frame['col0'][particle]) == 0):# and (particle%2 == 0):
                                            particle_pos_frame.append([
                                                round(float(data_frame['col1'][particle])/scale_dim, 6),
                                                round(float(data_frame['col2'][particle])/scale_dim, 6),
                                                round(float(data_frame['col3'][particle])/scale_dim, 6)])
                                    else:
                                        particle_pos_frame.append(particle_pos[frame-1][particle])
                                particle_pos.append(particle_pos_frame)
                                # Extra frame in the beginning
                                if frame == 241:
                                    particle_pos.append(particle_pos_frame.copy())
                                if (frame-240) % 10 == 0:
                                    print(c, ':', round((frame-240)/frames, 3)*100, '%')
                            particles = len(particle_pos[0])
    # Load PCA loading matrix and mean scalar
    with open(dataset_save + '/pca_eigen_vectors.pkl', 'rb') as f:
        W_wheel = pickle.load(f)
    with open(dataset_save + '/pca_mean_scalar.pkl', 'rb') as f:
        MEAN_wheel = pickle.load(f)
    with open(dataset_save + '/pca_eigen_values.pkl', 'rb') as f:
        V_wheel = pickle.load(f)
    # Convert 3D soil data to 2D data
    X2D_wheel = np.zeros((dimension*(frames+1), particles))
    for i in range((frames+1)):
        for j in range(particles):
            for k in range(dimension):
                X2D_wheel[k+i*dimension][j] = particle_pos[i][j][k] - MEAN_wheel

    modes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    mean_excav, std_excav = [], []
    mean_wheel, std_wheel = [], []
    for mode in modes:
        # Excav:
        Wr = W_excav[:, :mode+1]
        Xr2D = np.matmul(X2D_excav, Wr)
        X2D_after = np.matmul(Xr2D, np.transpose(Wr))
        # error = np.abs(np.subtract(X2D_excav, X2D_after))  # MAE
        error = np.power(np.subtract(X2D_excav, X2D_after), 2) # MSE
        if mode == 8:
            mean_particle_excav = np.mean(error, axis=0)
        error = error.flatten()
        mean_excav.append(np.mean(error))
        std_excav.append(np.std(error))

        # Wheel:
        Wr = W_wheel[:, :mode+1]
        Xr2D = np.matmul(X2D_wheel, Wr)
        X2D_after = np.matmul(Xr2D, np.transpose(Wr))
        # error = np.abs(np.subtract(X2D_wheel, X2D_after))  # MAE
        error = np.power(np.subtract(X2D_wheel, X2D_after), 2) # MSE
        if mode == 8:
            mean_particle_wheel = np.mean(error, axis=0)
        error = error.flatten()
        mean_wheel.append(np.mean(error))
        std_wheel.append(np.std(error))

    # Fig 1
    plt.figure(figsize=(15, 5))
    # plt.plot(np.multiply(np.subtract(modes, 10), 10), np.zeros(len(modes)), '--', c='gray', linewidth=1)
    plt.errorbar(modes, mean_excav, std_excav, fmt='--o', c='black', markersize=5, capsize=4)
    plt.errorbar(modes, mean_wheel, std_wheel, fmt='-.s', c='firebrick', markersize=5, capsize=4)
    ax = plt.gca()
    ax.set_xlim(min(modes)-20, max(modes)+20)
    # ax.set_ylim(bottom=1e-5)
    ax.set_xticks(modes)
    ax.set_yscale('log')
    ax.legend(['Excavation', 'Wheel'], fontsize=15)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.title('PCA Error', fontsize=15)
    plt.xlabel("#Modes", fontsize=15)
    plt.ylabel("Position MSE", fontsize=15)
    # plt.grid(b=True, which='minor', color='r', linestyle='--')
    plt.savefig("pca_error.png", dpi=300)

    # Fig 2
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.hist([mean_particle_excav, mean_particle_wheel], bins=1000, color=['black', 'firebrick'])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    plt.xlabel("Position MSE", fontsize=15)
    plt.ylabel("#Particles", fontsize=15)
    plt.title('PCA Error Distribution, 8 Modes', fontsize=15)
    ax.legend(['Excavation', 'Wheel'], loc="upper right", fontsize=15)
    plt.savefig("pca_error_distribution.png", dpi=300)

    # Fig 3
    dim = max(modes)
    V_excav_sum = np.sum(V_excav)
    V_wheel_sum = np.sum(V_wheel)
    energy_excav, energy_wheel = [], []
    for i in range(1, dim+1):
        energy_excav.append(np.sum(V_excav[:i]) / V_excav_sum)
        energy_wheel.append(np.sum(V_wheel[:i]) / V_wheel_sum)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, dim+1), energy_excav, '--o', c='black', markersize=5)
    plt.plot(range(1, dim+1), energy_wheel, '-.s', c='firebrick', markersize=5)
    ax = plt.gca()
    ax.set_xticks(modes)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.tick_params(axis='x', which='minor')
    # ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.xlabel("#Modes", fontsize=15)
    plt.ylabel("Normalized cumulative\nsum of eigenvalues", fontsize=15)
    plt.title('PCA Energy', fontsize=15)
    ax.legend(['Excavation', 'Wheel'], loc="lower right", fontsize=15)
    plt.savefig("pca_energy.png", dpi=300)