from matplotlib import animation
import numpy as np
import time
from src.diffadmm.diffadmm_wrapper import Deformable, DeformableConfig
from src.plotting import plot_init_rope, upd_plt_rope

cfg = DeformableConfig(
    "../data/rope/test_rope.yaml",
    dt=0.05,
    T=20,
    DEFAULT_BATCH_SIZE=1,
    gmres_m=150,
    gmres_restart=60,
    gmres_tol=1e-8,
    ADMM_Iter=1000,
    stretch_pen=1e3,
    bend_pen=0.0,
)

admm = Deformable(cfg)

B = 1
T = 20

pin_pos = np.array([[0, 0, 0], [0.1, 0.2, 0.3]])
pin_pos = np.repeat(pin_pos[None, ...], T, axis=0)
pin_pos = np.repeat(pin_pos[None, ...], B, axis=0)

t = [i for i in range(20)]


start = time.perf_counter()
J = admm.jxu(pin_pos, t)
end = time.perf_counter()
print(f"Jxu Calculation time: {end - start}")

start = time.perf_counter()
J = admm.jxu(pin_pos, t)
end = time.perf_counter()
print(f"Jxu Calculation time: {end - start}")


start = time.perf_counter()
fwd = admm.forwards(pin_pos)
end = time.perf_counter()
print(f"Forwards Calculation time: {end - start}")

# Plot

import matplotlib.pyplot as plt
import numpy as np

x = fwd["x_hist"][0, ...].copy()


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 1)
segs = plot_init_rope(ax, x)


def animate(t):
    upd_plt_rope(segs, x, t)
    return segs

ani = animation.FuncAnimation(
    fig, animate, frames=T, interval=50, blit=False, repeat=True  # 50ms = 0.05 seconds
)

plt.show() 
