from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


TYPE_TO_COLOR = {
    3: ["black", "o"],  # Boundary particles.
    0: ["green", "o"],  # Rigid solids.
    7: ["magenta", "o"],  # Goop.
    6: ["brown", "o"],  # Sand.
    5: ["blue", "o"],  # Water
    10: ["lightgray", "o"],  # added: Invis
}


def plot(data, type):
  num_steps = len(data)

  fig = plt.figure(figsize=plt.figaspect(0.5))
  ax = fig.add_subplot(1, 1, 1, projection='3d')

  ax.set_title('Fullspace Data')
  ax.set_xlim(-.5, .9)
  ax.set_ylim(-.5, 1)
  ax.set_zlim(-.5, .9)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  ax.view_init(elev=15, azim= 10)

  points = {
        particle_type: ax.plot([], [], [], value[1], ms=1/(particle_type+1)*10+0.5, color=value[0], zorder=1/(particle_type+1)*10)[0]
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
  unused_animation.save('fullspace.mp4', writer=writer, dpi=300)

  plt.show()
