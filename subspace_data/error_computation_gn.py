'''
Example usage (from parent directory):
`python -m subspace_data.error_computation_gn`

'''

import os
import pickle
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def main(data_path, rollout_path):
  with open(rollout_path, "rb") as file:
    rollout_data = pickle.load(file)

  # Load PCA loading matrix and mean scalar
  pca_load = data_path
  with open(os.path.join(pca_load, 'metadata.json'), 'rt') as fp:
    metadata = json.loads(fp.read())
  MODE_NUMBER = metadata["mode_number"]
  with open(os.path.join(pca_load, 'pca_eigen_vectors.pkl'), 'rb') as f:
    W = pickle.load(f)
    W = W[:, :MODE_NUMBER]
  with open(os.path.join(pca_load, 'pca_mean_scalar.pkl'), 'rb') as f:
    MEAN = pickle.load(f)

  c = 0
  for ax_i, (label, rollout_field) in enumerate(
      [(r"$\bf{Ground\ truth}$" + "\nMaterial point method", "ground_truth_rollout"),
       (r"$\bf{Prediction}$" + "\nGraph network", "predicted_rollout")]):
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]], axis=0)

    # Convert 3D soil data to 2D data
    particles_rigid = metadata["particles_rigid"]
    particles_nonrigid_full = W.shape[0]
    particles_all_sub = trajectory.shape[1]
    frames = trajectory.shape[0]
    dimension = trajectory.shape[2]
    Xr2D = np.zeros((trajectory.shape[2]*frames, particles_all_sub-particles_rigid))
    for i in range(frames):
      for j in range(particles_rigid, particles_all_sub):
        for k in range(dimension):
          Xr2D[k+i*dimension][j-particles_rigid] = trajectory[i][j][k]
    X2D = np.matmul(Xr2D, np.transpose(W))

    if c == 0:
      trajectory1 = X2D
    else:
      trajectory2 = X2D
    c += 1

  error = np.power(np.subtract(trajectory1, trajectory2), 2) # MSE
  mean_particle = np.mean(error, axis=0)
  return mean_particle


if __name__ == "__main__":
  data_path = 'learning_to_simulate/datasets/Excavation_PCA'
  rollout_path = 'learning_to_simulate/rollouts/Excavation_PCA/rollout_train_0_16modes.pkl'
  mean_particle_excav = main(data_path, rollout_path)

  data_path = 'learning_to_simulate/datasets/Wheel_PCA'
  rollout_path = 'learning_to_simulate/rollouts/Wheel_PCA/rollout_train_0.pkl'
  mean_particle_wheel = main(data_path, rollout_path)

  # Fig 1
  plt.figure(figsize=(10, 5))
  ax = plt.gca()
  ax.set_xscale('log')
  ax.set_yscale('log')
  plt.hist([mean_particle_excav, mean_particle_wheel], bins=1000, color=['black', 'firebrick'])
  ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
  plt.xlabel("Position MSE", fontsize=15)
  plt.ylabel("#Particles", fontsize=15)
  plt.title('GN Error Distribution, 8 Modes', fontsize=15)
  ax.legend(['Excavation', 'Wheel'], loc="upper right", fontsize=15)
  plt.savefig("gn_error_distribution.png", dpi=300)
