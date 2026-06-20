import sys

sys.path.insert(0, "/home/dshen/Code/deform-planning/src/diffadmm/build")

import numpy as np
import diffadmm

B = 2
N = 12
E = N - 1
L = 0.1
dt = 1e-2
T = 30
ADMM_ITERS = 50

x0 = np.zeros((B, 3 * N))
x0[:, 0::3] = np.arange(N) * L
v0 = np.zeros((B, 3 * N))

mass = np.ones((B, N))
L0 = np.full((B, E), L)
stiffness_stretch = np.full((B, E), 1e3)
penalty_stretch = np.full((B, E), 1e3)
stiffness_bend = np.full((B, N - 2), 1e2)
damping = np.full((B, N), 0.1)

pin_indices = [0, -1]
pin_velocities = np.zeros((len(pin_indices), 3))


sim = diffadmm.forward(
    x0=x0,
    v0=v0,
    mass=mass,
    L0=L0,
    stiffness_stretch=stiffness_stretch,
    penalty_stretch=penalty_stretch,
    stiffness_bend=stiffness_bend,
    damping=damping,
    pin_indices=pin_indices,
    pin_velocities=pin_velocities,
    dt=dt,
    penalty_bend=0.0,
    ADMM_ITERS=ADMM_ITERS,
    T=T,
    admm_tol=0.0,
    admm_check_interval=0,
    bending_admm=False,
    stretching_admm=True,
    bending_as_force=True,
)

for k, v in sim.items():
    print(f"{v.shape}")

xh = sim["x_hist"]

t = T // 2

J = np.asarray(
    diffadmm.compute_jxu(
        x_hist=sim["x_hist"],
        y_hist=sim["y_hist"],
        z_hist=sim["z_hist"],
        dual_hist=sim["dual_hist"],
        z_bend_hist=sim["z_bend_hist"],
        dual_bend_hist=sim["dual_bend_hist"],
        mass=mass,
        A_inv=sim["A_inv"],
        L0=L0,
        stiffness_stretch=stiffness_stretch,
        penalty_stretch=penalty_stretch,
        stiffness_bend=stiffness_bend,
        penaltyDt_stretch=sim["penaltyDt_stretch"],
        D_bend=sim["D_bend"],
        W_bend=sim["W_bend"],
        rest_curv=sim["rest_curv"],
        pin_indices=pin_indices,
        t=t,
        dt=dt,
        penalty_bend=0.0,
        gmres_m=120,
        gmres_restart=2,
        gmres_tol=5e-5,
        bending_admm=False,
        stretching_admm=True,
        bending_as_force=True,
    )
)

print(J)

J2 = diffadmm.forwards_with_jxu(
    x0=x0,
    v0=v0,
    mass=mass,
    L0=L0,
    stiffness_stretch=stiffness_stretch,
    penalty_stretch=penalty_stretch,
    stiffness_bend=stiffness_bend,
    damping=damping,
    pin_indices=pin_indices,
    pin_velocities=pin_velocities,
    dt=dt,
    penalty_bend=0.0,
    ADMM_ITERS=ADMM_ITERS,
    T=T,
    bending_admm=False,
    stretching_admm=True,
    bending_as_force=True,
    admm_tol=0.0,
    admm_check_interval=0,
    t=t,
    gmres_m=120,
    gmres_restart=2,
    gmres_tol=5e-5,
)


print(J2)

print((J == J2).all())