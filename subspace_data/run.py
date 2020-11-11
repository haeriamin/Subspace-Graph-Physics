'''
Example usage (from parent directory):
`python -m subspace_data.run`

Ref:
https://docs.python.org/3/library/struct.html
'''

import numpy as np
import struct
import pickle
# import warnings
# warnings.filterwarnings("ignore")

from subspace_data import tfrecord
from subspace_data import pca


if __name__ == '__main__':

    ''' Deserialize tfrecord file '''
    dataset_name = 'Sand'
    dataset_split = 'train'  # 'train', 'valid' or 'test'
    particle_type, particle_key, X3D = tfrecord.deserialize(
        dataset_name, dataset_split)
    timestep_number = len(X3D)
    print()
    print('# of timesteps:', timestep_number)
    particle_number = len(X3D[0])
    print('# of particles:', particle_number)
    dim_number = len(X3D[0][0])
    print('# of dimensions:', dim_number)
    print()

    ''' Process data '''
    MODE_NUMBER = 256

    # Convert 3D data to 2D data
    X2D = np.zeros((dim_number*timestep_number, particle_number))
    for i in range(timestep_number):
        for j in range(particle_number):
            for k in range(dim_number):
                X2D[dim_number*i+k][j] = X3D[i][j][k]

    # Apply PCA to 2D data
    idx, W, Y2D = pca.evd_np(X2D, MODE_NUMBER)

    # Save PCA transformation matrix, W
    data_path = './learning_to_simulate/datasets/' + dataset_name
    with open(data_path + '/' + dataset_split + '_pca_mat.pkl', 'wb') as f:
        pickle.dump(W, f)

    # Convert 2D data to 3D data
    Y3D = np.zeros((timestep_number, MODE_NUMBER, dim_number))
    for i in range(dim_number*timestep_number):
        for j in range(MODE_NUMBER):
            Y3D[i // int(dim_number)][j][i % int(dim_number)] = Y2D[i][j]

    # Convert 3D data to 2D data
    Y2D_2 = np.zeros((timestep_number, MODE_NUMBER*dim_number))
    for i in range(timestep_number):
        for j in range(MODE_NUMBER):
            for k in range(dim_number):
                Y2D_2[i][dim_number*j+k] = Y3D[i][j][k]

    ''' Serialize via tfrecord '''
    particle_type_sorted = []
    for i in range(MODE_NUMBER):
        particle_type_sorted.append(int(particle_type[idx[i]]))
    particle_type_bytes = struct.pack(
        '%sq' % len(particle_type_sorted), *particle_type_sorted)

    particle_position_bytes = []
    for i in range(timestep_number):
        particle_position_bytes.append(struct.pack(
            '%sf' % len(Y2D_2[i]), *Y2D_2[i]))

    subspace_data = {
        'context1': particle_type_bytes,
        'context2': particle_key,
        'feature': particle_position_bytes}

    tfrecord.serialize(subspace_data, dataset_name, dataset_split+'_sub')
