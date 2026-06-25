from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import sys
import pinocchio as pin

from src.diffadmm.diffadmm_wrapper import Deformable, DeformableConfig
from src.plotting import x_at_config, upd_plt_arm, gen_plt_from_urdf

"""
Tests forward dynamics to be used in the AGHF
"""

# --- Scenario setup ---

urdf_path = "src/data/urdf/kinova3_1arm.urdf"
rope_path = "src/data/rope/arm_rope_1pin.yaml"

dt = 0.05
T = 20

admm_kwargs = {
    "source": rope_path,
    "dt": dt,
    "T": T,
    "DEFAULT_BATCH_SIZE": 1,
    "gmres_m": 150,
    "gmres_restart": 60,
    "gmres_tol": 1e-08,
    "ADMM_Iter": 1000,
    "stretch_pen": 1000.0,
    "bend_pen": 0.0,
}

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

cfg = DeformableConfig(**admm_kwargs)
admm = Deformable(cfg)

# Pinned position
pin_frames = ["arm1_end_effector"]
pin_ids = [model.getFrameId(n) for n in pin_frames]


q0 = pin.neutral(model) 
v0 = pin.neutral(model) 

u = np.zeros((T, model.nv))
u[:, 0] = 0.1  # spin

model.gravity = pin.Motion.Zero()

# --- Parameter verification ---
# make sure pinned rope vertices and pinned control vertices coincide
pin_error_tol = 1e-2
# pin.framesForwardKinematics(model, data, q)

# if (len(pin_frames) != admm.r.n_u / 3):
#     raise ValueError(f"Inconsistent pinned number")

# for i, id in enumerate(pin_ids):
#     rope_pos = admm.r.x0[admm.r.pinned[i]]
#     manip_pos = data.oMf[id].translation
    
#     err = np.linalg.norm(rope_pos - manip_pos)

#     if err > pin_error_tol:
#         raise ValueError(f"Manipulator position ({manip_pos}) and rope starting position ({rope_pos}) are too far. Error: {err} > {pin_error_tol}")


# --- Compute forwards arm dynamics ---

q_a = np.zeros((T, model.nv))
v_a = np.zeros((T, model.nv))
a_a = np.zeros((T-1, model.nv)) # For logging purposes

q_a[0] = q0
v_a[0] = v0

H = np.zeros((T, model.nv, model.nv))
C = np.zeros((T, model.nv))
p_a_drift = np.zeros((T, len(pin_frames), 3))

for t in range(T - 1):

    pin.crba(model, data, q_a[t])
    H[t] = data.M
    M = data.M
    H[t] = np.triu(M) + np.triu(M, 1).T

    C[t] = pin.rnea(model, data, q_a[t], v_a[t], np.zeros(model.nv))

    a_a[t] = np.linalg.solve(H[t], u[t] - C[t])

    # Forward Euler, maybe make better?
    v_a[t + 1] = v_a[t] + dt * a_a[t]
    q_a[t + 1] = pin.integrate(model, q_a[t], dt * v_a[t + 1]) # Is semi-implicit valid for discrete time dynamics?




fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)
segs = gen_plt_from_urdf(ax, model, data, q0)

def animate(t):
    upd_plt_arm(segs, model, data, q_a[t])
    return segs

ani = animation.FuncAnimation(
    fig, animate, frames=T, interval=dt * 1e3, blit=False, repeat=True 
)

plt.show() 

# pin.computeJointJacobians(model, data, q)


# for i in range(1, model.njoints):    
#     J = pin.getJointJacobian(
#         model, 
#         data, 
#         i, 
#         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
#     )
    
#     print(f"{model.names[i]} {J}")

