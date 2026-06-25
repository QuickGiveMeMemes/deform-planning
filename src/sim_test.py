import time
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import sys
import pinocchio as pin

from src.diffadmm.diffadmm_wrapper import Deformable, DeformableConfig
from src.plotting import upd_plt_arm, gen_plt_from_urdf, plot_init_rope, upd_plt_rope

"""
Tests forward dynamics to be used in the AGHF
"""

# --- Scenario setup ---

urdf_path = "src/data/urdf/kinova3_1arm.urdf"
rope_path = "src/data/rope/arm_rope_1pin.yaml"

dt = 0.05
T = 40
B = 1

admm_kwargs = {
    "source": rope_path,
    "dt": dt,
    "T": T,
    "DEFAULT_BATCH_SIZE": B,
    "gmres_m": 150,
    "gmres_restart": 60,
    "gmres_tol": 1e-08,
    "ADMM_Iter": 2000,
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
n_pins = len(pin_frames)


q0 = pin.neutral(model) 
v0 = pin.neutral(model) 

u = np.ones((T, model.nv)) * 0
u[:, 0] = 0.05

model.gravity = pin.Motion.Zero() # Otherwise the arm collapses

# --- Parameter verification ---
# make sure pinned rope vertices and pinned control vertices coincide
# The sim can figure it out even without this, but the initial conditions 
# should be known instead of modified by the sim internally.

pin_error_tol = 1e-2
pin.framesForwardKinematics(model, data, q0)

if (n_pins != admm.r.n_u / 3):
    raise ValueError(f"Inconsistent pinned number")

for i, id in enumerate(pin_ids):
    rope_pos = admm.r.x0[admm.r.pinned[i]]
    manip_pos = data.oMf[id].translation
    
    err = np.linalg.norm(rope_pos - manip_pos)

    if err > pin_error_tol:
        raise ValueError(f"Manipulator position ({manip_pos}) and rope starting position ({rope_pos}) are too far. Error: {err} > {pin_error_tol}")


# --- Compute forwards arm dynamics ---
# 
# TODO maybe add batch dimension? not sure if it would be useful

q_a = np.zeros((T, model.nv))
v_a = np.zeros((T, model.nv))
a_a = np.zeros((T-1, model.nv)) # For logging purposes

q_a[0] = q0
v_a[0] = v0

H = np.zeros((T, model.nv, model.nv))
C = np.zeros((T, model.nv))

for t in range(T - 1):

    pin.crba(model, data, q_a[t])
    H[t] = data.M
    M = data.M
    H[t] = np.triu(M) + np.triu(M, 1).T

    C[t] = pin.rnea(model, data, q_a[t], v_a[t], np.zeros(model.nv))

    a_a[t] = np.linalg.solve(H[t], u[t] - C[t])

    # Forward Euler
    v_a[t + 1] = v_a[t] + dt * a_a[t]
    q_a[t + 1] = pin.integrate(model, q_a[t], dt * v_a[t + 1]) # Is semi-implicit valid for discrete time dynamics?

# Extracting pinned trajectory, other useful values from FK
p = np.zeros((B, T, n_pins, 3)) # Batch is hardcoded to 1
pdot = np.zeros((T, n_pins, 3))
p_a_drift = np.zeros((T, len(pin_frames), 3))
Jp = np.zeros((T, n_pins, 3, model.nv))

for t in range(T):
    pin.computeJointJacobians(model, data, q_a[t])
    pin.forwardKinematics(model, data, q_a[t], v_a[t], np.zeros(model.nv))
    pin.updateFramePlacements(model, data)

    for i in range(n_pins):
        p[0, t, i] = data.oMf[pin_ids[i]].translation

        Jp[t, i] = pin.getFrameJacobian(
            model, 
            data, 
            pin_ids[i], 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:3, :]

        pdot[t, i] = pin.getFrameVelocity(
            model,
            data,
            pin_ids[i],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear

        p_a_drift[t, i] = pin.getFrameClassicalAcceleration(
            model,
            data,
            pin_ids[i],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        

# Generating actual diffadmm forward sim rope trajectory

start = time.perf_counter()
fwd_out = admm.forwards(p)
end = time.perf_counter()
print(f"Forwards Calculation time: {end - start}")
x_fwd_out = fwd_out["x_hist"][0]

# Generating rope trajectory from Jxu extraction
t = [t for t in range(T)]
start = time.perf_counter()
jxp = admm.jxu(p, t).reshape((T, admm.r.N, 3, n_pins, 3))
end = time.perf_counter()
print(f"Jxu Calculation time: {end - start}")

jxp_red = jxp[:-1]
jxp_dot = (jxp[1:] - jxp_red) / dt

# p = Jp @ a_a + a_drift
# p:    (t-1, nu, 3)
# Jp:   (t-1, nu, 3, nv)
# q:    (t-1, nv)
pddot = np.einsum("tuav,tv->tua", Jp[:-1], a_a) + p_a_drift[:-1]

# a_r = jxp_dot @ pdot + jxp @ pddot
# a_r:              (t-1, N, 3)
# jxp_dot, jxp:     (t-1, N, 3, nu, 3)
# pdot, pddot:      (t-1, nu, 3)
a_r = np.einsum("tnaub,tub->tna", jxp_dot, pdot[:-1]) + np.einsum("tnaub,tub->tna", jxp_red, pddot)

# Currently only zero v initial
v_r = np.repeat(admm.r.v0[None, ...], T, axis=0)
v_r[1:] += np.cumsum(a_r, axis=0) * dt
q_r = np.repeat(admm.r.x0[None, ...], T, axis=0)
q_r[1:] += np.cumsum(v_r[1:], axis=0) * dt

print(q_r.shape)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 2)
segs_a = gen_plt_from_urdf(ax, model, data, q0)
segs_r_fwd_out = plot_init_rope(ax, x_fwd_out)
segs_r_jxu = plot_init_rope(ax, q_r)

def animate(t):
    upd_plt_arm(segs_a, model, data, q_a[t])
    upd_plt_rope(segs_r_fwd_out, x_fwd_out, t)
    upd_plt_rope(segs_r_jxu, q_r, t)

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

