from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


TYPE_TO_COLOR = {
    3: ["black", "o"],  # Boundary particles.
    0: ["black", "o"],  # Rigid solids.
    7: ["magenta", "o"],  # Goop.
    6: ["brown", "o"],  # Sand.
    5: ["blue", "o"],  # Water
    10: ["lightgray", "o"],  # added: Invis
}


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


def plot(data, type):
  num_steps = len(data)

  fig = plt.figure(figsize=plt.figaspect(0.5))
  ax = fig.add_subplot(1, 1, 1, projection='3d')

  ax.set_title('Subspace data')
  # ax.set_title('Fullspace data')
  # ax.set_xlim(-.5, .9)
  # ax.set_ylim(-.5, 1)
  # ax.set_zlim(-.5, .9)
  ax.set_xlim(0, 0.5)
  ax.set_ylim(0, 0.5)
  ax.set_zlim(0, 0.5)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  set_axes_equal(ax)
  ax.view_init(elev=0, azim=10)

  points = {
        # particle_type: ax.plot([], [], [], value[1], ms=1/(particle_type+1)*10+0.5, color=value[0], zorder=1/(particle_type+1)*10)[0]  # excav
        particle_type: ax.plot([], [], [], value[1], ms=1/(particle_type+3)*10+0.5, color=value[0], zorder=1/(particle_type+1)*10)[0]  # wheel
        for particle_type, value in TYPE_TO_COLOR.items()}

  def update(frame):
    outputs = []
    for particle_type, line in points.items():
      mask = type[frame] == particle_type
      line.set_data(data[frame][mask, 2], data[frame][mask, 0])
      line.set_3d_properties(data[frame][mask, 1])
      outputs.append(line)
    return outputs

  unused_animation = animation.FuncAnimation(fig, update, frames=np.arange(0, num_steps), interval=10)

  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'mpeg4'])
  unused_animation.save('subspace_data.mp4', writer=writer, dpi=300)

  plt.show()
