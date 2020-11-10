'''
Example usage (from parent directory):
`python -m subspace_data.run`

Refs:
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
https://aihub.cloud.google.com/u/0/p/products%2Fcd31a10b-85db-4c78-830c-a3794dd606ce
https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

'''

import numpy as np

from subspace_data import tfrecord
from subspace_data import pca


if __name__ == '__main__':

    ''' Deserialize tfrecord file '''
    dataset_name = 'Sand'
    split = 'train'  # 'train', 'valid' or 'test'
    X3D = tfrecord.deserialize(dataset_name, split)
    timestep_number = len(X3D)
    # print(timestep_number)
    particle_number = len(X3D[0])
    # print(particle_number)
    dim_number = len(X3D[0][0])
    # print(dim_number)

    ''' Process data '''
    MODE_NUMBER = 256
    # Convert 3D data to 2D data
    X2D = np.zeros((2*timestep_number, particle_number))
    for i in range(timestep_number):
        for j in range(particle_number):
            for k in range(dim_number):
                X2D[2*i+k][j] = X3D[i][j][k]
    # Apply PCA to 2D data
    id, W, Y2D = pca.evd_np(X2D, MODE_NUMBER)
    # Convert 2D data to 3D data
    Y3D = np.zeros((timestep_number, MODE_NUMBER, dim_number))
    for i in range(2*timestep_number):
        for j in range(MODE_NUMBER):
            Y3D[i // 2][j][i % 2] = Y2D[i][j]
    # print(id)