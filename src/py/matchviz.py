"""
matchviz.py - solution diagnostics for the ADMM rope-manipulation pipeline,
with a YAML-driven POINT-MATCHING target overlay.

The LEAP decision here is ARM-ONLY (sol CSV = [q; v; a; u] x Nn, nfree=0); the
rope is reconstructed by the diffADMM rollout, so run the cpp dump
(PointMatchCost::dumpRope) to emit `sol_kinova_rope_rope.csv` (T x 3N, uniform grid).

Target: the objective is now POINT MATCHING, not full-shape matching. The
scenario yaml carries

    track_idx:  [1, 4, 8]              # rope vertex indices
    track_targ: [[x,y,z], ...]         # one world-space target per tracked idx

so there is no goal rope to overlay. Tracked vertices get circle markers on the
simulated rope; their targets get X markers. Pinned vertices (the EE attachment
points) keep their squares.
"""

import argparse
import json
import os
from pathlib import Path

from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

try:
    from yaml import safe_load
except Exception:
    safe_load = None

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
SCENARIO = "./src/data/problems/2arm1rope_match.yaml"
META = "sol_kinova_rope_alm_meta.json"
SOL = "sol_kinova_rope_alm_mode0.csv"
CONS = "sol_kinova_rope_alm_cons_mode0.csv"
ROPE = "sol_kinova_rope_rope.csv"

URDF = "./src/data/urdf/kinova3_2arm.urdf"
FRAMES = ["arm1_end_effector_link", "arm2_end_effector_link"]
T_HORIZON = 2.0

GHOST_KW = dict(alpha=0.35)
TRACK_C = "tab:red"      # tracked vertices + their targets
PIN_C = "tab:orange"     # pinned (EE) vertices


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
def load_scenario(path):
    """Goal + geometry from the scenario yaml. Returns None if unavailable."""
    if safe_load is None or not os.path.exists(path):
        print(f"[scenario] {path} unavailable - target overlay disabled")
        return None
    with open(path) as f:
        y = safe_load(f)

    track_idx = list(y.get("track_idx", []))
    track_targ = np.asarray(y["track_targ"], float) if y.get("track_targ") else None
    if track_targ is not None:
        track_targ = track_targ.reshape(-1, 3)
        if len(track_idx) != len(track_targ):
            raise ValueError(
                f"track_idx ({len(track_idx)}) and track_targ ({len(track_targ)}) "
                "length mismatch"
            )

    sc = dict(
        urdf=y.get("urdf", URDF),
        frames=y.get("pinFrames", FRAMES),
        T=float(y.get("T", T_HORIZON)),
        q0_arm=np.asarray(y["q0_arm"], float),
        qf_arm=np.asarray(y["qf_arm"], float) if y.get("qf_arm") else None,
        q0_rope=np.asarray(y["q0_rope"], float),
        track_idx=track_idx,
        track_targ=track_targ,
        rope_yaml=y.get("ropeYaml"),
    )

    pinned, rest = None, None
    if sc["rope_yaml"] and os.path.exists(sc["rope_yaml"]):
        with open(sc["rope_yaml"]) as f:
            ry = safe_load(f)
        pinned = list(ry.get("control", {}).get("pinned", []))
        verts = ry.get("vertices")
        if verts:
            V = np.asarray(verts, float)
            rest = np.linalg.norm(np.diff(V, axis=0), axis=1)   # per-EDGE rest length
    sc["pinned"] = pinned if pinned else [0, len(sc["q0_rope"]) - 1]
    sc["rest"] = rest

    print(f"[scenario] {path}: pinned={sc['pinned']}  track_idx={track_idx}")
    if rest is None:
        print("[scenario] no rope yaml vertices - strain falls back to |x(0)| edges")
    return sc


def load_meta(path):
    with open(path) as f:
        m = json.load(f)
    mode = m["modes"][0]
    return dict(nq=m["nq"], nv=m["nv"], joint_names=m["joint_names"],
                Nn=mode["Nn"], degree=mode["degree"], T=mode["T"],
                dims=mode["dims"], taus=np.asarray(mode["taus"], float))


def load_sol(path, meta):
    Z = np.genfromtxt(path, delimiter=",")            # (nP, Nn)
    nP, Nn = Z.shape
    d = meta["dims"]
    nq, nv, na, nu = d["nq"], d["nv"], d["na"], d.get("nu", 0)
    assert nq + nv + na + nu == nP, f"{nq}+{nv}+{na}+{nu} != nP={nP}"
    q = Z[:nq]; v = Z[nq:nq + nv]; a = Z[nq + nv:nq + nv + na]
    u = Z[nq + nv + na:nq + nv + na + nu] if nu else None
    t = 0.5 * meta["T"] * (1.0 + meta["taus"])
    print(f"[sol] nP={nP} Nn={Nn}  q/v/a/u = {nq}/{nv}/{na}/{nu}")
    return t, meta["taus"], q.T, v.T, a.T, (u.T if u is not None else None)


def load_cons(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    node = rows["node"].astype(int)
    return node, {n: rows[n] for n in rows.dtype.names if n != "node"}


def load_rope(path):
    if not os.path.exists(path):
        print(f"[rope] {path} not found - run the cpp dump to enable rope panels")
        return None
    with open(path) as f:
        hdr = f.readline()
    m = {}
    for tok in hdr.lstrip("#").split():
        k, _, val = tok.partition("=")
        m[k] = val
    T, N, dt = int(m["T"]), int(m["N"]), float(m["dt"])
    pinned = [int(x) for x in m.get("pinned", "").split(";") if x != ""]
    X = np.genfromtxt(path, delimiter=",", skip_header=1).reshape(T, N, 3)
    print(f"[rope] T={T} N={N} dt={dt} pinned={pinned}")
    return dict(t=np.arange(T) * dt, R=X, dt=dt, pinned=pinned, N=N)


# ---------------------------------------------------------------- CGL bary interp
def cgl_bary_weights(n):
    w = np.ones(n); w[1::2] = -1.0; w[0] *= 0.5; w[-1] *= 0.5
    return w


def bary_interp(tau_nodes, F_nodes, tau_query, w=None):
    n = len(tau_nodes)
    w = cgl_bary_weights(n) if w is None else w
    out = np.empty((len(tau_query), F_nodes.shape[1]))
    for m, tq in enumerate(tau_query):
        diff = tq - tau_nodes
        hit = np.isclose(diff, 0.0, atol=1e-14)
        if hit.any():
            out[m] = F_nodes[np.argmax(hit)]
        else:
            c = w / diff
            out[m] = (c[:, None] * F_nodes).sum(0) / c.sum()
    return out


# ---------------------------------------------------------------- plots
def plot_constraints(node, groups):
    fig, ax = plt.subplots(figsize=(9, 4))
    floor = 1e-16
    for name, vals in groups.items():
        v = np.asarray(vals, float)
        if np.nanmax(np.abs(v)) <= floor:
            continue
        ax.semilogy(node, np.abs(v) + floor, "o-", ms=3, lw=1, label=name)
    ax.set_xlabel("node"); ax.set_ylabel("|violation|")
    ax.set_title("CONSTRAINT VIOLATIONS by group")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()


def plot_arm(t, q, v, a, u, names, qf_goal=None):
    npan = 4 if u is not None else 3
    fig, ax = plt.subplots(npan, 1, figsize=(9, 3 * npan), sharex=True)
    labels = [n.replace("arm", "a").replace("_joint_", "j") for n in names]
    for j in range(q.shape[1]):
        (ln,) = ax[0].plot(t, q[:, j], "o-", ms=3, lw=1, label=labels[j])
        if qf_goal is not None:
            ax[0].axhline(qf_goal[j], color=ln.get_color(), ls="--", lw=0.8, **GHOST_KW)
    gtxt = "  (dashed = qf ref from yaml)" if qf_goal is not None else ""
    ax[0].set_ylabel("q [rad]"); ax[0].set_title("ARM JOINTS" + gtxt)
    ax[0].legend(fontsize=6, ncol=7)
    for j in range(v.shape[1]):
        ax[1].plot(t, v[:, j], "-", lw=1)
    ax[1].set_ylabel("v [rad/s]"); ax[1].set_title("ARM JOINT VEL")
    for j in range(a.shape[1]):
        ax[2].plot(t, a[:, j], "-", lw=1)
    ax[2].set_ylabel("a [rad/s^2]"); ax[2].set_title("ARM JOINT ACCEL")
    if u is not None:
        for j in range(u.shape[1]):
            ax[3].plot(t, u[:, j], "-", lw=1)
        ax[3].set_ylabel("u [N.m]"); ax[3].set_title("ARM TORQUE (inverse dynamics)")
    ax[-1].set_xlabel("t [s]")
    fig.tight_layout()


def plot_rope(rope, track_idx=None, track_targ=None, rest=None):
    """Strain / height / tracking-error panels. Tracking error replaces the old
    goal-rope overlay: with point matching there is no full goal shape."""
    t, R, pinned = rope["t"], rope["R"], rope["pinned"]
    edge = np.linalg.norm(np.diff(R, axis=1), axis=2)        # (T, N-1)
    if rest is None or len(rest) != edge.shape[1]:
        rest = np.linalg.norm(np.diff(R[0], axis=0), axis=1)  # fallback: t=0 edges
        rest_src = "x(0)"
    else:
        rest_src = "rope yaml"
    strain = edge / rest[None, :]

    has_track = track_idx is not None and track_targ is not None and len(track_idx)
    npan = 3 if has_track else 2
    fig, ax = plt.subplots(npan, 1, figsize=(9, 3.4 * npan), sharex=True)

    # --- strain -------------------------------------------------------------
    for i in range(strain.shape[1]):
        ax[0].plot(t, strain[:, i], "o-", ms=3, lw=1, label=f"e{i}")
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_ylabel("len / rest"); ax[0].set_yscale("log")
    ax[0].set_title(f"ROPE STRAIN  max={strain.max():.3f}x  min={strain.min():.3f}x"
                    f"  (rest from {rest_src}: {rest.min():.3f}-{rest.max():.3f} m)")
    ax[0].legend(fontsize=7, ncol=5)

    # --- height -------------------------------------------------------------
    ax[1].plot(t, R[:, :, 2].min(axis=1), "-", lw=1, color="gray", label="lowest vertex")
    for v in pinned:                                    # EE / pinned keep SQUARES
        ax[1].plot(t, R[:, v, 2], "s-", ms=4, lw=1, color=PIN_C, label=f"pin v{v}")
    if has_track:
        for j, v in enumerate(track_idx):               # tracked get CIRCLES
            (ln,) = ax[1].plot(t, R[:, v, 2], "o-", ms=3.5, lw=1, label=f"track v{v}")
            ax[1].axhline(track_targ[j, 2], color=ln.get_color(), ls="--", lw=0.9,
                          **GHOST_KW)
    ax[1].axhline(0, color="k", ls=":", lw=1)
    ax[1].set_ylabel("z [m]")
    ax[1].set_title("ROPE HEIGHT  (squares = pinned/EE, circles = tracked, "
                    "dashed = target z)")
    ax[1].legend(fontsize=8, ncol=3)

    # --- tracking error -----------------------------------------------------
    err = None
    if has_track:
        err = np.linalg.norm(R[:, track_idx, :] - track_targ[None, :, :], axis=2)
        for j, v in enumerate(track_idx):
            ax[2].semilogy(t, err[:, j] + 1e-16, "o-", ms=3.5, lw=1, label=f"v{v}")
        ax[2].semilogy(t, err.max(axis=1) + 1e-16, "k--", lw=1.2, label="max")
        ax[2].set_ylabel("|x_v - target| [m]"); ax[2].set_xlabel("t [s]")
        ax[2].set_title("POINT-MATCH TRACKING ERROR   "
                        f"final max={err[-1].max():.4f} m  "
                        f"(best over horizon {err.max(axis=1).min():.4f} m)")
        ax[2].grid(True, which="both", alpha=0.3)
        ax[2].legend(fontsize=8, ncol=4)
    else:
        ax[1].set_xlabel("t [s]")

    fig.tight_layout()
    return strain, err


def animate(model, data, rope, q_u, sc=None, track_idx=None, track_targ=None, fps=8):
    try:
        from src.py.plotting import gen_plt_from_urdf, upd_plt_arm
    except Exception as e:
        print(f"[anim] src.py.plotting unavailable ({e}); skipping animation")
        return None
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    t, R = rope["t"], rope["R"]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    segs = gen_plt_from_urdf(ax, model, data, q_u[0], seg_kwargs={"color": "k"})
    (rope_ln,) = ax.plot3D(*R[0].T, "-", lw=1.5, color="tab:blue", label="rope")

    pins = rope["pinned"]
    (pin_pts,) = ax.plot3D(*R[0, pins].T, "s", ms=6, color=PIN_C, label="pins (EE)")

    all_pts = [R.reshape(-1, 3)]
    trk_pts = None
    has_track = track_idx is not None and track_targ is not None and len(track_idx)
    if has_track:
        (trk_pts,) = ax.plot3D(*R[0, track_idx].T, "o", ms=6, color=TRACK_C,
                               label="tracked")
        ax.plot3D(*track_targ.T, "X", ms=10, color=TRACK_C, mew=1.5,
                  label="targets", **GHOST_KW)
        all_pts.append(track_targ)

    arm_lo = np.array([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]])
    arm_hi = np.array([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])
    P = np.vstack(all_pts)
    lo = np.minimum(arm_lo, P.min(0))
    hi = np.maximum(arm_hi, P.max(0))
    c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.1
    for setl, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setl(c[i] - r, c[i] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    ax.legend(fontsize=8); ttl = ax.set_title("")

    def upd(k):
        upd_plt_arm(segs, model, data, q_u[k])
        rope_ln.set_data_3d(*R[k].T)
        pin_pts.set_data_3d(*R[k, pins].T)
        if trk_pts is not None:
            trk_pts.set_data_3d(*R[k, track_idx].T)
        ttl.set_text(f"frame {k}/{len(t)-1}   t={t[k]:.3f}s   (X = point-match targets)")

    return FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADMM rope solution diagnostics.")
    parser.add_argument("--cfid", type=str, help="scenario yaml id")
    args = parser.parse_args()
    if args.cfid is not None:
        SCENARIO = Path("./src/data/problems/generated") / f"2arm1rope_{args.cfid}.yaml"

    meta = load_meta(META)
    t, tau, q, v, a, u = load_sol(SOL, meta)
    node, groups = load_cons(CONS)
    rope = load_rope(ROPE)
    sc = load_scenario(SCENARIO)

    plot_constraints(node, groups)
    plot_arm(t, q, v, a, u, meta["joint_names"],
             qf_goal=(sc["qf_arm"] if sc else None))

    track_idx = sc["track_idx"] if sc else None
    track_targ = sc["track_targ"] if sc else None
    rest = sc["rest"] if sc else None

    model = data = None
    try:
        import pinocchio as pin
        model = pin.buildModelFromUrdf(sc["urdf"] if sc else URDF)
        data = model.createData()
    except Exception as e:
        print(f"[goal] pinocchio/URDF unavailable ({e}); 3D animation disabled")

    anim = None
    if rope is not None:
        if track_idx:
            bad = [i for i in track_idx if not (0 <= i < rope["N"])]
            if bad:
                raise ValueError(f"track_idx {bad} out of range for N={rope['N']}")
        s, err = plot_rope(rope, track_idx, track_targ, rest)
        print(f"[diag] rope strain {s.min():.3f}x .. {s.max():.3f}x")
        if err is not None:
            for j, vtx in enumerate(track_idx):
                print(f"[diag] track v{vtx}: final |e|={err[-1, j]:.4f} m   "
                      f"min over horizon={err[:, j].min():.4f} m")
        if model is not None:
            tau_q = 2.0 * rope["t"] / meta["T"] - 1.0
            q_u = bary_interp(tau, q, tau_q)
            anim = animate(model, data, rope, q_u, sc=sc,
                           track_idx=track_idx, track_targ=track_targ)
    savefolder = Path("./out/gifs")
    savefolder.mkdir(parents=True, exist_ok=True)
    anim.save(savefolder / "match_out.mp4", writer=animation.FFMpegWriter(fps=30))
    plt.show()