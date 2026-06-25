from matplotlib import animation
import numpy as np
import time
from diffadmm_wrapper import Deformable, DeformableConfig

cfg = DeformableConfig(
    "../data/rope/test_rope.yaml",
    dt=0.05,
    T=20,
    seg=0.2,
    sim_time=1.0,
    eps_fd=1e-8,
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


def plot_init(ax, x: np.ndarray):
    # Assuming dims (T, 3N) so batch already removed
    T = x.shape[0]
    N = x.shape[1] // 3

    x = x.reshape((T, N, 3))

    segs = []
    for _ in range(1, N):
        (seg,) = ax.plot3D(x[:, 0], x[:, 1], x[:, 2])
        segs.append(seg)

    ax.set_axis_on()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return segs


def upd_plt(segs, x, t):

    T = x.shape[0]
    N = x.shape[1] // 3

    x = x.reshape((T, N, 3))

    for i, seg in enumerate(segs):
        seg.set_data_3d(
            np.vstack(
                (
                    x[t, i, :],
                    x[t, i + 1, :],
                )
            ).T
        )


x = fwd["x_hist"][0, ...].copy() # whyyy ...


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 1)
segs = plot_init(ax, x)


def animate(t):
    # This function is called automatically for every frame
    upd_plt(segs, x, t)
    return segs


# Set up the animation to play through the frames (0 to T-1)
ani = animation.FuncAnimation(
    fig, animate, frames=T, interval=50, blit=False, repeat=True  # 50ms = 0.05 seconds
)

plt.show()  # This takes over the event loop safely
