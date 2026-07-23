"""
ropeviz.py — solution diagnostics for the ADMM rope-manipulation pipeline,
with a YAML-driven GOAL-POSE overlay.
 
The LEAP decision here is ARM-ONLY (sol CSV = [q; v; a; u] x Nn, nfree=0); the
rope is reconstructed by the diffADMM rollout, so run the cpp dump
(RopeCost::dumpRope) to emit `sol_kinova_rope_rope.csv` (T x 3N, uniform grid).
 
Goal pose: the solver's terminal state is NOT the target — the pin/boundary
constraints are penalties, not hard equalities, so the solved endpoint can drift
from the requested goal. We therefore read the goal straight from the scenario
YAML (`qf_arm`, and the target rope) and render it as a semitransparent overlay:
  * 3D: ghost arm (via FK) + ghost rope at qf.
  * static panels: dashed goal lines (final arm q, and per-vertex rope strain/z).
 
Target rope: if `qf_rope_enable` is true, use `qf_rope` verbatim; otherwise
reconstruct it the way the cpp does — the initial rope rigidly translated by the
FK displacement of pin 0's frame between q0 and qf (see rope_admm_opt.cpp).
"""
 
import argparse
import json
import os
from pathlib import Path
 
import numpy as np
import matplotlib.pyplot as plt
 
try:
    from yaml import safe_load
except Exception:
    safe_load = None
 
# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
SCENARIO = "./src/data/problems/generated/2arm1rope_lift.yaml"               # scenario yaml (source of the GOAL)
META = "sol_kinova_rope_alm_meta.json"
SOL = "sol_kinova_rope_alm_mode0.csv"
CONS = "sol_kinova_rope_alm_cons_mode0.csv"
ROPE = "sol_kinova_rope_rope.csv"            # from RopeCost::dumpRope (optional)
 
URDF = "./src/data/urdf/kinova3_2arm.urdf"   # fallbacks if scenario yaml missing
FRAMES = ["arm1_end_effector_link", "arm2_end_effector_link"]
REST = 0.1
T_HORIZON = 2.0
 
GHOST_KW = dict(alpha=0.35)                   # semitransparent overlay style
 
 
# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
def load_scenario(path):
    """Goal + geometry from the scenario yaml. Returns None if unavailable."""
    if safe_load is None or not os.path.exists(path):
        print(f"[scenario] {path} unavailable — goal overlay disabled")
        return None
    with open(path) as f:
        y = safe_load(f)
    sc = dict(
        urdf=y.get("urdf", URDF),
        frames=y.get("pinFrames", FRAMES),
        T=float(y.get("T", T_HORIZON)),
        q0_arm=np.asarray(y["q0_arm"], float),
        qf_arm=np.asarray(y["qf_arm"], float),
        q0_rope=np.asarray(y["q0_rope"], float),          # (N,3)
        qf_rope_enable=bool(y.get("qf_rope_enable", False)),
        qf_rope=np.asarray(y["qf_rope"], float) if y.get("qf_rope") else None,
        rope_yaml=y.get("ropeYaml"),
    )
    pinned = None
    if sc["rope_yaml"] and os.path.exists(sc["rope_yaml"]):
        with open(sc["rope_yaml"]) as f:
            ry = safe_load(f)
        pinned = list(ry.get("control", {}).get("pinned", []))
    sc["pinned"] = pinned if pinned else [0, len(sc["q0_rope"]) - 1]
    print(f"[scenario] {path}: goal qf_arm loaded, pinned={sc['pinned']}, "
          f"qf_rope_enable={sc['qf_rope_enable']}")
    return sc
 
 
def goal_rope(sc, model, data):
    """Target rope at qf. Either qf_rope verbatim, or x0_rope rigidly translated
    by pin-0 frame's FK displacement q0->qf (matches rope_admm_opt.cpp)."""
    if sc["qf_rope_enable"] and sc["qf_rope"] is not None:
        return sc["qf_rope"]
    import pinocchio as pin
    fid = model.getFrameId(sc["frames"][0])
 
    def fkpos(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[fid].translation)
 
    disp = fkpos(sc["qf_arm"]) - fkpos(sc["q0_arm"])
    return sc["q0_rope"] + disp[None, :]
 
 
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
        print(f"[rope] {path} not found — run the cpp dump to enable rope panels")
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
    gtxt = "  (dashed = qf goal from yaml)" if qf_goal is not None else ""
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
 
 
def plot_rope(rope, goal_R=None):
    t, R, pinned = rope["t"], rope["R"], rope["pinned"]
    strain = np.linalg.norm(np.diff(R, axis=1), axis=2) / REST
    rest_meas = np.linalg.norm(np.diff(R[0], axis=0), axis=1)
 
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for i in range(strain.shape[1]):
        ax[0].plot(t, strain[:, i], "o-", ms=3, lw=1, label=f"e{i}")
    if goal_R is not None:
        gs = np.linalg.norm(np.diff(goal_R, axis=0), axis=1) / REST
        for i, val in enumerate(gs):
            ax[0].axhline(val, ls="--", lw=0.8, color=f"C{i % 10}", **GHOST_KW)
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_ylabel("len / rest"); ax[0].set_yscale("log")
    gtxt = "  (dashed = goal-rope strain)" if goal_R is not None else ""
    ax[0].set_title(f"ROPE STRAIN  max={strain.max():.2f}x  min={strain.min():.2f}x"
                    f"  (rest {rest_meas.min():.3f}-{rest_meas.max():.3f} m)" + gtxt)
    ax[0].legend(fontsize=7, ncol=5)
 
    ax[1].plot(t, R[:, :, 2].min(axis=1), "o-", ms=3, label="lowest vertex")
    for v in pinned:
        ax[1].plot(t, R[:, v, 2], "o-", ms=3, label=f"pin v{v}")
    if goal_R is not None:
        ax[1].axhline(goal_R[:, 2].min(), ls="--", lw=0.9, color="gray",
                      label="goal min-z", **GHOST_KW)
    ax[1].axhline(0, color="k", ls=":", lw=1)
    ax[1].set_ylabel("z [m]"); ax[1].set_title("ROPE HEIGHT"); ax[1].legend(fontsize=8)
 
    az = np.gradient(np.gradient(R[:, :, 2], rope["dt"], axis=0), rope["dt"], axis=0)
    for i in range(az.shape[1]):
        ax[2].plot(t, az[:, i], "-", lw=1)
    ax[2].axhline(-9.81, color="k", ls="--", lw=1, label="-g (free fall)")
    ax[2].set_ylabel("a_z [m/s^2]"); ax[2].set_xlabel("t [s]")
    ax[2].set_title("ROPE VERTICAL ACCEL"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    return strain
 
 
def animate(model, data, rope, q_u, sc=None, goal_R=None, fps=8):
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

    arm_kwargs = {
        "color": "k"
    }
 
    segs = gen_plt_from_urdf(ax, model, data, q_u[0], seg_kwargs=arm_kwargs)
    (rope_ln,) = ax.plot3D(*R[0].T, "-", ms=4, lw=1.5, color="tab:blue", label="rope")
    pins = rope["pinned"]
    (pin_pts,) = ax.plot3D(*R[0, pins].T, "s", ms=5, color="tab:orange", label="pins")
 
    # ---- semitransparent GOAL overlay (arm ghost + rope ghost) at qf ----
    all_pts = [R.reshape(-1, 3)]
    # if sc is not None:
    #     try:
    #         ghost_segs = gen_plt_from_urdf(ax, model, data, sc["qf_arm"])
    #         for s in ghost_segs:
    #             try:
    #                 s.set_alpha(GHOST_KW["alpha"]); s.set_color("gray")
    #             except Exception:
    #                 pass
    #     except Exception as e:
    #         print(f"[anim] ghost arm failed ({e})")
    if goal_R is not None:
        ax.plot3D(*goal_R.T, "o--", ms=3, lw=1.2, color="tab:blue",
                  label="rope goal", **GHOST_KW)
        all_pts.append(goal_R)
 
    arm_lo = np.array([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]])
    arm_hi = np.array([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])
    P = np.vstack(all_pts)
    rope_lo = P.min(0)
    rope_hi = P.max(0)
    lo = np.minimum(arm_lo, rope_lo)
    hi = np.maximum(arm_hi, rope_hi)
    c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.1
    for setl, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setl(c[i] - r, c[i] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass # Ignore for older matplotlib versions
    ax.legend(fontsize=8); ttl = ax.set_title("")
 
    def upd(k):
        upd_plt_arm(segs, model, data, q_u[k])
        rope_ln.set_data_3d(*R[k].T)
        pin_pts.set_data_3d(*R[k, pins].T)
        ttl.set_text(f"frame {k}/{len(t)-1}   t={t[k]:.3f}s   (ghost = qf goal)")
 
    return FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps)
 
 
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to process custom user data.")   
    parser.add_argument("--cfid", type=str, help="yaml")

    args = parser.parse_args()
    if args.cfid is not None:
        SCENARIO = Path("./src/data/problems/generated") / f"2arm1rope_{args.cfid}.yaml"

    meta = load_meta(META)
    t, tau, q, v, a, u = load_sol(SOL, meta)
    node, groups = load_cons(CONS)
    rope = load_rope(ROPE)
    sc = load_scenario(SCENARIO)
 
    qf_goal = sc["qf_arm"] if sc is not None else None
    plot_constraints(node, groups)
    plot_arm(t, q, v, a, u, meta["joint_names"], qf_goal=qf_goal)
 
    goal_R = None
    model = data = None
    try:
        import pinocchio as pin
        model = pin.buildModelFromUrdf(sc["urdf"] if sc else URDF)
        data = model.createData()
        if sc is not None:
            goal_R = goal_rope(sc, model, data)
    except Exception as e:
        print(f"[goal] pinocchio/URDF unavailable ({e}); "
              "goal rope overlay uses qf_rope only if provided")
        if sc is not None and sc["qf_rope_enable"]:
            goal_R = sc["qf_rope"]
 
    anim = None
    if rope is not None:
        s = plot_rope(rope, goal_R=goal_R)
        print(f"[diag] rope strain {s.min():.2f}x .. {s.max():.2f}x")
        if model is not None:
            tau_q = 2.0 * rope["t"] / meta["T"] - 1.0
            q_u = bary_interp(tau, q, tau_q)
            anim = animate(model, data, rope, q_u, sc=sc, goal_R=goal_R)
 
    plt.show()