import json
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.py.plotting import gen_plt_from_urdf, upd_plt_arm

URDF   = "./src/data/urdf/kinova3_1arm.urdf"
CSV    = "sol_kinova_rope_alm_mode0.csv"
FRAME  = "arm1_end_effector_link"
PINNED = [0]
T      = 3.0
REST   = 0.1          # rest edge length from the yaml

def load(csv, nq_arm, T):
    """CSV[i][k] = z_i at node k, z = [q; v; a]. Returns t, q_arm, xr, vr, ar."""
    Z  = np.genfromtxt(csv, delimiter=",")        # (nP, Nn)
    nP, Nn = Z.shape
    nq = nP // 3                                   # nlam = nu = 0
    q, v, a = Z[:nq], Z[nq:2*nq], Z[2*nq:]
    nfree = (nq - nq_arm) // 3
    # CGL nodes ascending in time: tau_k = -cos(pi k / (Nn-1))
    tau = -np.cos(np.pi * np.arange(Nn) / (Nn - 1))
    t   = 0.5 * T * (1.0 + tau)
    rs  = lambda F: F[nq_arm:].reshape(nfree, 3, Nn).transpose(2, 0, 1)  # (Nn,nfree,3)
    print(f"[schema] nP={nP} Nn={Nn} nq={nq} nq_arm={nq_arm} nfree={nfree}")
    return t, q[:nq_arm].T, rs(q), rs(v), rs(a)

def frame_pos(model, data, q, name):
    fid = model.getFrameId(name)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.array(data.oMf[fid].translation)

def full_rope(model, data, q_arm, xr, name, pinned):
    Nn, nfree, _ = xr.shape
    nvert = nfree + len(pinned)
    free = [v for v in range(nvert) if v not in pinned]
    R = np.zeros((Nn, nvert, 3))
    for k in range(Nn):
        R[k, free] = xr[k]
        p = frame_pos(model, data, q_arm[k], name)
        for v in pinned:
            R[k, v] = p
    return R

def diagnose(t, q_arm, R, ar, rest):
    seg    = np.linalg.norm(np.diff(R, axis=1), axis=2)
    strain = seg / rest

    fig, ax = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    for i in range(strain.shape[1]):
        ax[0].plot(t, strain[:, i], "o-", ms=3, lw=1, label=f"e{i}")
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_ylabel("len / rest"); ax[0].set_yscale("log")
    ax[0].set_title(f"STRAIN  max={strain.max():.1f}x  min={strain.min():.2f}x")
    ax[0].legend(fontsize=7, ncol=5)

    ax[1].plot(t, R[:, :, 2].min(axis=1), "o-", ms=3, label="lowest vertex")
    ax[1].plot(t, R[:, 0, 2], "o-", ms=3, label="pin (FK)")
    ax[1].axhline(0, color="k", ls=":", lw=1, label="floor")
    ax[1].set_ylabel("z [m]"); ax[1].set_title("HEIGHT"); ax[1].legend(fontsize=8)

    for j in range(q_arm.shape[1]):
        ax[2].plot(t, q_arm[:, j], "o-", ms=3, lw=1, label=f"j{j}")
    ax[2].set_ylabel("q [rad]"); ax[2].set_title("ARM JOINTS")
    ax[2].legend(fontsize=7, ncol=7)

    ax[3].plot(t, ar[:, :, 2], "o-", ms=3, lw=1)
    ax[3].axhline(-9.81, color="k", ls="--", lw=1, label="-g (free fall)")
    ax[3].set_ylabel("a_r,z [m/s^2]"); ax[3].set_xlabel("t [s]")
    ax[3].set_title("ROPE VERTICAL ACCEL"); ax[3].legend(fontsize=8)
    fig.tight_layout()
    return strain

def animate(model, data, t, q_arm, R, fps=8):
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_axis_on()
    ax.autoscale(False)


    segs = gen_plt_from_urdf(ax, model, data, q_arm[0])
    (rope,) = ax.plot3D(*R[0].T, "o-", ms=4, lw=1.5, color="tab:red", label="rope")
    (pt,)   = ax.plot3D(*R[0, :1].T, "*", ms=14, color="tab:green", label="pin")
    lo, hi = R.reshape(-1, 3).min(0), R.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.1
    for setl, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setl(c[i] - r, c[i] + r)
    ax.legend(fontsize=8); ttl = ax.set_title("")
    def upd(k):
        upd_plt_arm(segs, model, data, q_arm[k])
        rope.set_data_3d(*R[k].T); pt.set_data_3d(*R[k, :1].T)
        ttl.set_text(f"node {k}/{len(t)-1}   t={t[k]:.3f}s")
    return FuncAnimation(fig, upd, frames=len(t), interval=1000/fps)

if __name__ == "__main__":
    model = pin.buildModelFromUrdf(URDF); data = model.createData()
    t, q_arm, xr, vr, ar = load(CSV, model.nq, T)
    R = full_rope(model, data, q_arm, xr, FRAME, PINNED)
    s = diagnose(t, q_arm, R, ar, REST)
    print(f"[diag] strain     {s.min():.2f}x .. {s.max():.2f}x")
    print(f"[diag] z range    {R[:,:,2].min():+.2f} .. {R[:,:,2].max():+.2f} m")
    print(f"[diag] |q_arm|max {np.abs(q_arm).max():.2f} rad")
    an = animate(model, data, t, q_arm, R)
    plt.show()