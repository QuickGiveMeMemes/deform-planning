import time
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import sys
import pinocchio as pin
from scipy.interpolate import interp1d
from scipy.sparse import csr_array

from src.py.plotting import upd_plt_arm, gen_plt_from_urdf

"""
The AGHF becomes least squares
"""

urdf_path = "src/data/urdf/kinova3_1arm.urdf"

dt = 0.01
T = 200

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

q0 = pin.neutral(model) 
v0 = np.zeros(model.nv)

qf = np.array(
    [
        4.21283162,
        1.6307425,
        1.03610551,
        0.81460703,
        1.52834486,
        1.01943345,
        1.25715633,
    ]
)
vf = np.zeros(model.nv)

x0 = np.hstack((q0, v0))[:, None]
xf = np.hstack((qf, vf))[:, None]

print(x0.shape)

interp = interp1d(np.array([0, T * dt]), np.hstack((x0, xf)).T, axis=0) 
x_init = interp(np.arange(0, T) * dt)

x_state = x_init.flatten()

H0 = pin.crba(model, data, q0)

# T -> 0 to T-1

def residual(x_state, k=1e3):

    P = np.zeros((T - 1) * model.nv * 2)
    
    for t in range(T - 1):
        offset = t * model.nv * 2
        qt = x_state[offset : offset + model.nv]
        vt = x_state[offset + model.nv : offset + 2 * model.nv]
        qt_next = x_state[offset + 2 * model.nv : offset + 3 * model.nv]
        vt_next = x_state[offset + 3 * model.nv : offset + 4 * model.nv]

        P[offset : offset + model.nv] = np.sqrt(k) * H0 @ ((qt_next - qt) / dt - vt) / dt
        P[offset + model.nv : offset + 2 * model.nv] = np.sqrt(k) * pin.rnea(model, data, qt, vt, (vt_next - vt) / dt)

    return P

def jacobian(x_state):
    dPtdxt = np.zeros((2 * model.nv, 2 * model.nv))
    dPtdxt_next = np.zeros((2 * model.nv, 2 * model.nv))

    block = [[None for _ in range(T - 2)] for _ in range(T - 1)]

    for t in range(T - 3):
        
        dPtdxt

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.set_ylim(-1, 1)
# ax.set_xlim(-1, 2)
# segs_a = gen_plt_from_urdf(ax, model, data, q0)

# def animate(t):
#     upd_plt_arm(segs_a, model, data, x_init[t, :model.nv])

# ani = animation.FuncAnimation(
#     fig, animate, frames=T, interval=dt * 1e3, blit=False, repeat=True 
# )
# plt.show() 