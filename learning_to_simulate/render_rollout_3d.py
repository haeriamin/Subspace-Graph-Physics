"""Simple matplotlib rendering of a rollout prediction against ground truth.
Usage (from parent directory):
`python -m learning_to_simulate.render_rollout_pca --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`
Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.
"""  # pylint: disable=line-too-long

import os
import pickle
import json

from absl import app
from absl import flags
import datetime

from celluloid import Camera
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_boolean("fullspace", True, help="full vs sub")
flags.DEFINE_string("data_path", None, help="Path to PCA files for postprocessing")
flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main(unused_argv):
  if not FLAGS.rollout_path:
    raise ValueError("A `rollout_path` must be passed.")
  with open(FLAGS.rollout_path, "rb") as file:
    rollout_data = pickle.load(file)

  # Load PCA loading matrix and mean scalar
  if not FLAGS.data_path:
    raise ValueError("A `data_path` must be passed.")
  pca_load = FLAGS.data_path
  with open(os.path.join(pca_load, 'metadata.json'), 'rt') as fp:
    metadata = json.loads(fp.read())
  MODE_NUMBER = 8#metadata["mode_number"]
  with open(os.path.join(pca_load, 'pca_eigen_vectors.pkl'), 'rb') as f:
    W = pickle.load(f)
    W = W[:, :MODE_NUMBER]
  with open(os.path.join(pca_load, 'pca_mean_scalar.pkl'), 'rb') as f:
    MEAN = pickle.load(f)


  # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  # ax = fig.add_subplot(projection='3d')

  fig = plt.figure(figsize=plt.figaspect(0.5))
  plt.subplots_adjust(wspace=0)

  plot_info = []
  for ax_i, (label, rollout_field) in enumerate(
      [(r"$\bf{Ground\ truth}$" + "\nMaterial point method", "ground_truth_rollout"),
       (r"$\bf{Prediction}$" + "\nGraph network", "predicted_rollout")]):
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field]], axis=0)


    # Convert 3D soil data to 2D data
    if FLAGS.fullspace:
      particles_rigid = metadata["particles_rigid"] #rollout_data["metadata"]["particles_rigid"]  #15
      particles_nonrigid_full = W.shape[0]
      particles_all_sub = trajectory.shape[1]
      frames = trajectory.shape[0]
      dimension = trajectory.shape[2]
      Xr2D = np.zeros((trajectory.shape[2]*frames, particles_all_sub-particles_rigid))
      for i in range(frames):
        for j in range(particles_rigid, particles_all_sub):
          for k in range(dimension):
            Xr2D[k+i*dimension][j-particles_rigid] = trajectory[i][j][k]
      # Apply PCA to 2D data
      X2D = np.matmul(Xr2D, np.transpose(W))
      # Convert 2D data to 3D data
      X = np.zeros((frames, particles_nonrigid_full+particles_rigid, dimension))
      for i in range(frames):
        for j in range(particles_rigid):
          for k in range(dimension):
            X[i][j][k] = trajectory[i][j][k]
      for i in range(dimension*frames):
        for j in range(particles_nonrigid_full):
          X[i // int(dimension)][j+particles_rigid][i % int(dimension)] = X2D[i][j] + MEAN
      trajectory = X


    ax = fig.add_subplot(1, 2, ax_i+1, projection='3d')
    ax.set_title(label)
    # bounds = rollout_data["metadata"]["bounds"]
    # ax.set_xlim(bounds[0][0], bounds[0][1])
    # ax.set_ylim(bounds[1][0], bounds[1][1])

    if 'Excavation' in FLAGS.rollout_path:
      # ax.set_xlim(0.1, 0.9)
      ax.set_xlim(0, 0.9)
      ax.set_ylim(0.2, 1.0)
      ax.set_zlim(0.1, 0.9)
    elif 'Wheel' in FLAGS.rollout_path:
      ax.set_xlim(0, 0.5)
      ax.set_ylim(0, 0.5)
      ax.set_zlim(0, 0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    set_axes_equal(ax)
    ax.view_init(elev=0, azim= 10)
    plot_info.append((ax, trajectory))


  # Create particle type full data
  if FLAGS.fullspace:
    for _ in range(particles_nonrigid_full+particles_rigid-particles_all_sub):
      rollout_data["particle_types"] = np.append(rollout_data["particle_types"], 6)


  # Animate
  num_steps = trajectory.shape[0]
  camera = Camera(fig)
  c = 0
  for ax, trajectory in plot_info:
    if c == 0:
      ax1 = ax
      trajectory1 = trajectory
    else:
      ax2 = ax
      trajectory2 = trajectory
    c += 1

  for i in range(num_steps):
    if i == 0:
      velocities1 = np.zeros(len(trajectory1[0]))
      velocities2 = np.zeros(len(trajectory2[0]))
    else:
      velocities1 = np.linalg.norm(np.subtract(trajectory1[i,:], trajectory1[i-1,:]), axis=1)/metadata["dt"]
      velocities2 = np.linalg.norm(np.subtract(trajectory2[i,:], trajectory2[i-1,:]), axis=1)/metadata["dt"]

    mask = rollout_data["particle_types"] == 6 #particle_type

    f = ax1.scatter(trajectory1[i, mask, 2], trajectory1[i, mask, 0], trajectory1[i, mask, 1], c=velocities1[mask], s=10, cmap='RdGy')#, norm=normalize)
    ax1.scatter(
      trajectory1[i, :metadata["particles_rigid"],2], trajectory1[i, :metadata["particles_rigid"], 0], trajectory1[i, :metadata["particles_rigid"], 1], c='Black', s=50)
    f = ax2.scatter(trajectory2[i, mask, 2], trajectory2[i, mask, 0], trajectory2[i, mask, 1], c=velocities2[mask], s=10, cmap='RdGy')
    ax2.scatter(
      trajectory2[i, :metadata["particles_rigid"], 2], trajectory2[i, :metadata["particles_rigid"], 0], trajectory2[i, :metadata["particles_rigid"], 1], c='Black', s=50)

    if not FLAGS.fullspace:
      ax1.plot3D(trajectory1[i, mask, 2], trajectory1[i, mask, 0], trajectory1[i, mask, 1], 'black')
      ax2.plot3D(trajectory2[i, mask, 2], trajectory2[i, mask, 0], trajectory2[i, mask, 1], 'black')

    camera.snap()

  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.85, 0.265, 0.025, 0.46])
  cbar = fig.colorbar(f, cax=cbar_ax)
  cbar.ax.get_yaxis().labelpad = 15
  if FLAGS.fullspace:
    cbar.ax.set_ylabel('Velocity magnitude [m/s]', rotation=270)
  else:
    cbar.ax.set_ylabel('Velocity magnitude (subspace)', rotation=270)

  anim = camera.animate(blit=True, interval=15)
  now = datetime.datetime.now()
  anim.save(str(now.strftime("%Y-%m-%d %H:%M:%S"))+'.mp4', dpi=300)


if __name__ == "__main__":
  app.run(main)
