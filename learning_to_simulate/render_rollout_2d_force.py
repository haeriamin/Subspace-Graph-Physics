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

flags.DEFINE_string("plane", None, help="xy, xz, or zy")
flags.DEFINE_string("data_path", None, help="Path to PCA files for postprocessing")
flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_integer("step_stride", 1, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS


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
  MODE_NUMBER = metadata["mode_number"]
  with open(os.path.join(pca_load, 'pca_eigen_vectors.pkl'), 'rb') as f:
    W = pickle.load(f)
    W = W[:, :MODE_NUMBER]
  with open(os.path.join(pca_load, 'pca_mean_scalar.pkl'), 'rb') as f:
    MEAN = pickle.load(f)


  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  plot_info = []
  for ax_i, (label, rollout_field) in enumerate(
      [(r"$\bf{Ground\ truth}$" + "\nMaterial point method", "ground_truth_"),
       (r"$\bf{Prediction}$" + "\nGraph network", "predicted_")]):
    trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data[rollout_field + 'rollout']], axis=0)

    forces = np.concatenate([
        rollout_data["initial_forces"],
        rollout_data[rollout_field + 'forces']], axis=0)

    # Convert 3D soil data to 2D data
    particles_all_sub = trajectory.shape[1]
    particles_nonrigid_full = W.shape[0]
    particles_rigid = particles_all_sub-MODE_NUMBER #rollout_data["metadata"]["particles_rigid"]
    # print(particles_nonrigid_full)
    # print(particles_rigid)
    # print(particles_all_sub)

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


    ax = axes[ax_i]
    ax.set_title(label)
    # bounds = rollout_data["metadata"]["bounds"]
    # ax.set_xlim(bounds[0][0], bounds[0][1])
    # ax.set_ylim(bounds[1][0], bounds[1][1])

    if 'Excavation' in FLAGS.rollout_path:
      if FLAGS.plane == 'xy':
        ax.set_xlim(0.1, 0.9)
        # ax.set_xlim(0.1, 1.2)
        ax.set_ylim(0.1, 0.7)
      elif FLAGS.plane == 'xz':
        ax.set_xlim(0.1, 0.9)
        ax.set_ylim(0.0, 0.8)
      elif FLAGS.plane == 'zy':
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(0.1, 0.7)
    elif 'Wheel' in FLAGS.rollout_path:
      if FLAGS.plane == 'xy':
        ax.set_xlim(0.1, 0.5)
        ax.set_ylim(0.1, 0.4)
      elif FLAGS.plane == 'xz':
        ax.set_xlim(0.1, 0.5)
        ax.set_ylim(0.0, 0.4)
      elif FLAGS.plane == 'zy':
        ax.set_xlim(0.0, 0.4)
        ax.set_ylim(0.1, 0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('equal', adjustable='box')
    ax.set_aspect(1.)
    plot_info.append((ax, trajectory, forces))


  # Create particle type full data
  for _ in range(particles_nonrigid_full+particles_rigid-particles_all_sub):
    rollout_data["particle_types"] = np.append(rollout_data["particle_types"], 6)


  # Animate
  num_steps = trajectory.shape[0]
  camera = Camera(fig)
  c = 0
  for ax, trajectory, forces in plot_info:
    if c == 0:
      ax1 = ax
      trajectory1 = trajectory
      forces1 = forces
      # print(forces1)
    else:
      ax2 = ax
      trajectory2 = trajectory
      forces2 = forces
      # print(forces2)
    c += 1

  velocities_max = np.linalg.norm(
    np.subtract(trajectory1[int(num_steps/2),metadata["particles_rigid"]], trajectory1[int(num_steps/2)-1,metadata["particles_rigid"]]))

  norm = matplotlib.colors.Normalize()
  cm = matplotlib.cm.winter
  sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
  sm.set_array([])

  for i in range(num_steps):
    mask = np.ones(len(trajectory1[0]))

    if i == 0:
      velocities1 = np.zeros(len(trajectory1[0]))
      velocities2 = np.zeros(len(trajectory2[0]))
    else:
      velocities1 = np.linalg.norm(np.subtract(trajectory1[i,:], trajectory1[i-1,:]), axis=1)/metadata["dt"]
      velocities2 = np.linalg.norm(np.subtract(trajectory2[i,:], trajectory2[i-1,:]), axis=1)/metadata["dt"]
    
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=1.5*velocities_max)

    # Get cross section in z direction:
    mask = rollout_data["particle_types"] == 6 #particle_type
    # for j in range(len(trajectory1[0])):
    #   if trajectory[i, j, 2] > 0.4:  # Excav: 0.4, Wheel: 0.2
    #     mask[j] = False

    if FLAGS.plane == 'xy':
      f = ax1.scatter(trajectory1[i, mask, 0], trajectory1[i, mask, 1], c=velocities1[mask], s=10, cmap='RdGy')#, norm=normalize)
      ax1.scatter(trajectory1[i,:metadata["particles_rigid"],0], trajectory1[i,:metadata["particles_rigid"],1], c='Black', s=50)
      ax1.plot(trajectory1[i,:metadata["particles_rigid"],0], trajectory1[i,:metadata["particles_rigid"],1], c='Black')
      ax1.quiver(
        np.mean(trajectory1[i,:metadata["particles_rigid"],0]), np.mean(trajectory1[i,:metadata["particles_rigid"],1]),
        forces1[i,0], forces1[i,1], color=cm(norm(forces1[i]))) #, scale=5e3, scale_units='xy', units='xy'
      
      f = ax2.scatter(trajectory2[i, mask, 0], trajectory2[i, mask, 1], c=velocities2[mask], s=10, cmap='RdGy')
      ax2.scatter(trajectory2[i,:metadata["particles_rigid"],0], trajectory2[i,:metadata["particles_rigid"],1], c='Black', s=50)
      ax2.plot(trajectory2[i,:metadata["particles_rigid"],0], trajectory2[i,:metadata["particles_rigid"],1], c='Black')
      ax2.quiver(
        np.mean(trajectory2[i,:metadata["particles_rigid"],0]), np.mean(trajectory2[i,:metadata["particles_rigid"],1]),
        forces2[i,0], forces2[i,1], color=cm(norm(forces2[i])))#, scale=5e3, scale_units='xy', units='xy'

    elif FLAGS.plane == 'xz':
      f = ax1.scatter(trajectory1[i, mask, 0], trajectory1[i, mask, 2], c=velocities1[mask], s=10, cmap='RdGy')#, norm=normalize)
      ax1.scatter(trajectory1[i,:metadata["particles_rigid"],0], trajectory1[i,:metadata["particles_rigid"],2], c='Black', s=50)
      f = ax2.scatter(trajectory2[i, mask, 0], trajectory2[i, mask, 2], c=velocities2[mask], s=10, cmap='RdGy')
      ax2.scatter(trajectory2[i,:metadata["particles_rigid"],0], trajectory2[i,:metadata["particles_rigid"],2], c='Black', s=50)

    elif FLAGS.plane == 'zy':
      f = ax1.scatter(trajectory1[i, mask, 2], trajectory1[i, mask, 1], c=velocities1[mask], s=10, cmap='RdGy')#, norm=normalize)
      ax1.scatter(trajectory1[i,:metadata["particles_rigid"],2], trajectory1[i,:metadata["particles_rigid"],1], c='Black', s=50)
      f = ax2.scatter(trajectory2[i, mask, 2], trajectory2[i, mask, 1], c=velocities2[mask], s=10, cmap='RdGy')
      ax2.scatter(trajectory2[i,:metadata["particles_rigid"],2], trajectory2[i,:metadata["particles_rigid"],1], c='Black', s=50)
    
    camera.snap()

  fig.subplots_adjust(right=0.87, wspace=0.025)

  cbar_ax = fig.add_axes([0.878, 0.22, 0.02, 0.55])
  cbar = fig.colorbar(f, cax=cbar_ax)
  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('Velocity magnitude [m/s]', rotation=270)

  cbar_ax = fig.add_axes([0.097, 0.22, 0.02, 0.55])
  cbar = fig.colorbar(sm, cax=cbar_ax, ticklocation='left')
  cbar.ax.get_yaxis().labelpad = 15
  cbar.ax.set_ylabel('Force magnitude [N]', rotation=90)
  # cbar.ax.yaxis.set_ticks_position('left')

  anim = camera.animate(blit=True, interval=15)
  #now = datetime.datetime.now()
  #anim.save(str(now.strftime("%Y-%m-%d %H:%M:%S"))+'.mp4', dpi=300)
  #anim.save(FLAGS.rollout_path + '.mp4', dpi=300)
  fname = FLAGS.rollout_path.split(".pkl")[0] + "_2D_Force_" + FLAGS.plane + ".mp4"
  anim.save(fname, dpi=300)

if __name__ == "__main__":
  app.run(main)
