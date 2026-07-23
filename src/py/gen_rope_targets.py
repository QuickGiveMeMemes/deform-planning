"""
gen_rope_cases.py — ROPE-FIRST test-case generator for the 2-arm rope problem.

Inverted architecture. Previously the rope was derived from the arms: FK(q0) gave
two endpoints and the rope was whatever arch fit between them, so rope length was
an implicit consequence of `sag` and could not be set directly. Here the ROPE is
the primary object --

    length [m], segments, shape (Bezier/helix control points), position, rotation

-- and the arms are solved to it by IK. That makes rope length, discretization and
shape independent knobs, which is what you want when sweeping cases.

Guarantees, by construction rather than by luck:

  * EXACT pin consistency. IK never lands perfectly, so the rope is REBUILT between
    the ACHIEVED end-effector positions after IK. Pin vertices therefore equal
    FK(q0) to machine precision and the solver's pin-vs-frame check cannot fire.

  * EXACT length. Rebuilding between fixed endpoints would change arc length, so a
    bulge scale alpha is solved (bisection, arc length is monotone in alpha) to
    restore the requested length. Both start and goal end up at the same length.

  * ZERO stretch at the goal. Both shapes are sampled at uniform arc length, so
    every edge is length/segments in both, and per-edge strain is ~0. At ks=1e6 a
    goal that is 1% long is one the stretch term resists with ~1e4 N -- the
    tracking cost can never reach it, and the resulting floor looks like a solver
    pathology when it is a geometry error.

  * The goal is REACHABLE. The goal rope is likewise rebuilt between the IK-achieved
    end-effector positions, so it is not asking for endpoints the arms cannot hold.

Shapes are cubic-or-higher Bezier curves given by interior control points in a
normalized chord frame (x along the chord in (0,1); y lateral; z up, so sag is
NEGATIVE z), plus an analytic helix. Presets cover sag / S / W / hook / loop /
twist / coil; `Shape(ctrl=...)` is the escape hatch for anything else.

Run from repo root (needs pinocchio + the URDF).
  python gen_rope_cases.py                    # all cases
  python gen_rope_cases.py sag_to_taut loop   # a subset, by name
  python gen_rope_cases.py --preview          # also write a 3D png per case
"""

from pathlib import Path
import sys
from dataclasses import dataclass, field, replace
from math import comb
from typing import Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------- config
URDF = "./src/data/urdf/kinova3_2arm.urdf"
FRAMES = ["arm1_end_effector_link", "arm2_end_effector_link"]

# IK seed (7 DOF per arm). Only a starting point -- IK moves off it.
Q_SEED = [0.35, -0.45, 0.25, 0.30, 0.0, 0.0, 0.0,
          -0.30, 0.55, -0.20, -0.35, 0.0, 0.0, 0.0]

T_HORIZON = 2.0
DIFFADMM_STEPS = 21

# Spectral degree -> N_n = deg+1 nodes. The stretch boundary layer sets a
# resolution floor; deg=16 (N_n=17) sat marginally below it in earlier runs and
# read as "poor convergence" when it was really under-resolution.
DEG = 20

ROPE_DIR = "./src/data/rope/generated"
SCEN_DIR = "./src/data/problems/generated"
PREVIEW_DIR = "./out/preview"

ROPE_MASS = 0.1          # per-vertex (the loader expands per-vertex, not per-DOF)
ROPE_DAMPING = 0.1
ROPE_KS = 1e4
ROPE_KB = 1e1

MIN_EE_SEP = 0.15        # two Kinova EEs cannot occupy the same space [m]
MAX_EE_TRAVEL = 0.60     # endpoint displacement the arms must cover in T [m]
FLOOR_Z = 0.0            # warn if any vertex dips below this
IK_TOL = 1e-9
IK_WARN_RES = 1e-4       # above this the placement is effectively unreachable

M_DENSE = 4001           # densification before arc-length resampling
WORLD_UP = np.array([0.0, 0.0, 1.0])
CHORD_AXIS = np.array([0.0, 1.0, 0.0])   # default: rope spans left-right (+/- y)


# ---------------------------------------------------------------- shape spec
@dataclass(frozen=True)
class Shape:
    """Normalized shape. x runs along the chord in (0,1); y is lateral; z is UP,
    so a hanging rope has NEGATIVE z control points. Magnitudes are relative to
    the chord, so a shape is scale-free -- physical size comes from `length`."""
    kind: str = "bezier"                       # "bezier" | "helix" | "catenary"
    ctrl: Tuple[Tuple[float, float, float], ...] = ()   # interior control points
    turns: float = 1.0                         # helix only
    radius: float = 0.12                       # helix only
    slack: float = 1.0                         # catenary only: arc/chord at placement

    def nominal_alpha(self):
        """alpha used to size the chord before IK. For a catenary alpha IS the
        arc/chord ratio, so it must be the requested slack -- at alpha=1 a catenary
        is a straight line, which places the endpoints a full rope-length apart and
        leaves nothing to hang."""
        return self.slack if self.kind == "catenary" else 1.0


# presets -- "dynamic" shapes beyond a plain arch
def SAG(d=0.35):        return Shape(ctrl=((0.33, 0.0, -d), (0.67, 0.0, -d)))
def CATENARY(slack=1.20): return Shape(kind="catenary", slack=slack)
def TAUT():             return Shape(ctrl=((0.33, 0.0, -0.02), (0.67, 0.0, -0.02)))
def S_WAVE(a=0.25):     return Shape(ctrl=((0.30, 0.0, -a), (0.70, 0.0, a)))
def W_WAVE(a=0.22):     return Shape(ctrl=((0.25, 0.0, -a), (0.50, 0.0, a), (0.75, 0.0, -a)))
def HOOK(d=0.55):       return Shape(ctrl=((0.20, 0.0, -d), (0.80, 0.0, -0.15 * d)))
def LOOP(d=0.65):       return Shape(ctrl=((0.85, 0.0, -d), (0.15, 0.0, -d)))
def TWIST(a=0.30):      return Shape(ctrl=((0.30, a, -a), (0.70, -a, -a)))
def COIL(turns=1.0, r=0.12): return Shape(kind="helix", turns=turns, radius=r)


# ---------------------------------------------------------------- case table
@dataclass
class Case:
    name: str
    length: float                       # rope rest length [m]  <-- primary knob
    segments: int                       # edges; vertices = segments + 1
    start: Shape
    goal: Shape
    mid: Tuple[float, float, float] = (0.55, 0.0, 0.60)   # chord midpoint [m]
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)     # chord frame orientation
    goal_mid: Optional[Tuple[float, float, float]] = None  # defaults to `mid`
    goal_rpy: Optional[Tuple[float, float, float]] = None  # defaults to `rpy`

    def goal_placement(self):
        return (self.goal_mid or self.mid, self.goal_rpy or self.rpy)


CASES = [
    # Starts are CATENARY where possible: with a straight rest shape and v0=0, a
    # non-equilibrium start swings, and that transient runs through the whole
    # tracking horizon. Goals are free-form -- nothing says a goal is at rest.
    # --- shape change in place: same length, same location, different curve
    Case("sag_to_taut",  length=0.75, segments=9,  start=CATENARY(), goal=TAUT()),
    Case("taut_to_sag",  length=0.75, segments=9,  start=TAUT(),     goal=SAG(0.45)),
    Case("sag_to_s",     length=0.75, segments=11, start=CATENARY(), goal=S_WAVE(0.28)),
    Case("s_to_w",       length=0.85, segments=13, start=S_WAVE(0.25), goal=W_WAVE(0.24)),
    Case("sag_to_hook",  length=0.80, segments=11, start=CATENARY(), goal=HOOK(0.55)),
    # --- out of plane
    Case("sag_to_twist", length=0.80, segments=11, start=CATENARY(), goal=TWIST(0.30)),
    Case("coil",         length=0.95, segments=15, start=CATENARY(), goal=COIL(1.0, 0.14)),
    # --- large deformation
    Case("sag_to_loop",  length=0.95, segments=15, start=CATENARY(), goal=LOOP(0.65)),
    # --- transport: shape fixed, placement moves (isolates travel from deformation)
    Case("lift",         length=0.75, segments=9,  start=CATENARY(), goal=CATENARY(),
         goal_mid=(0.55, 0.0, 0.74)),
    Case("shift",        length=0.75, segments=9,  start=CATENARY(), goal=CATENARY(),
         goal_mid=(0.68, 0.0, 0.56)),
    Case("rotate",       length=0.75, segments=9,  start=CATENARY(), goal=CATENARY(),
         goal_rpy=(0.0, 0.0, 0.45)),
    # --- resolution sweep: identical physics, different discretization
    Case("sag_n6",       length=0.75, segments=6,  start=CATENARY(), goal=S_WAVE(0.28)),
    Case("sag_n20",      length=0.75, segments=20, start=CATENARY(), goal=S_WAVE(0.28)),
    
    Case(
        "custom1", 
        length=0.55, 
        segments=9, 
        start=CATENARY(), 
        goal=Shape(
            ctrl=((0.20, -0.05, -0.50,), (0.50, -0.05, -0.40), (0.80, 0.05, 0.1)),
        ), 
        goal_mid=(0.48, 0.0, 0.56),
        goal_rpy=(1.0, 0.0, 0.85),
    ),
]


# ---------------------------------------------------------------- curve maths
def bezier(P, m=M_DENSE):
    """Bernstein evaluation for an arbitrary number of control points."""
    P = np.asarray(P, float)
    n = len(P) - 1
    t = np.linspace(0.0, 1.0, m)[:, None]
    out = np.zeros((m, 3))
    for i, Pi in enumerate(P):
        b = comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i))
        out += b * Pi[None, :]
    return out


def arclen(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def resample_arclen(P, n):
    """n vertices at equal arc-length spacing (preview/diagnostics only -- NOT how
    the rope is discretized; see equal_chord_polygon)."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    t = np.linspace(0.0, cs[-1], n)
    return np.stack([np.interp(t, cs, P[:, k]) for k in range(3)], axis=1)


def _step_chord(P, i0, c, e):
    """First point along polyline P after index i0 at Euclidean distance e from c."""
    for i in range(i0 + 1, len(P)):
        if np.linalg.norm(P[i] - c) >= e:
            a, b = P[i - 1], P[i]
            d, f = b - a, a - c
            A_ = float(d @ d)
            if A_ <= 0.0:
                continue
            B_, C_ = 2.0 * float(f @ d), float(f @ f) - e * e
            disc = max(B_ * B_ - 4.0 * A_ * C_, 0.0)
            t = min(max((-B_ + np.sqrt(disc)) / (2.0 * A_), 0.0), 1.0)
            return a + t * d, i - 1
    return None, len(P) - 1


def equal_chord_polygon(P, e, k):
    """Inscribe a k-edge polygon with EVERY EDGE exactly e, starting at P[0].

    Equal arc length is the wrong discretization here: the simulated rope is a
    polyline, so its rest lengths are CHORDS, and a chord across a high-curvature
    stretch is much shorter than its arc (2R sin(t/2) vs R t). Sampling at equal
    arc length therefore leaves edges that vary with curvature -- measured at up to
    1.6% on the sine cases and 27% on the loop, which at ks=1e6 is a goal the
    stretch term will never allow. Marching equal chords makes rest lengths uniform
    by construction. Returns (vertices, last_index) or (None, _) if P is too short."""
    pts, c, i = [P[0]], P[0], 0
    for _ in range(k):
        c, i = _step_chord(P, i, c, e)
        if c is None:
            return None, i
        pts.append(c)
    return np.asarray(pts), i


def rodrigues(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rpy_matrix(rpy):
    r, p, y = rpy
    return (rodrigues(WORLD_UP, y)
            @ rodrigues(np.array([0.0, 1.0, 0.0]), p)
            @ rodrigues(np.array([1.0, 0.0, 0.0]), r))


def chord_frame(A, B, up_hint=WORLD_UP):
    """Columns [e1 e2 e3]: e1 along the chord, e3 the in-plane up."""
    e1 = B - A
    d = np.linalg.norm(e1)
    e1 = e1 / d
    u = up_hint - (up_hint @ e1) * e1
    if np.linalg.norm(u) < 1e-8:
        u = np.cross(e1, np.array([1.0, 0.0, 0.0]))
    e3 = u / np.linalg.norm(u)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=1), d


def _catenary(A, B, s, m=M_DENSE):
    """Dense catenary from A to B with arc length s -- the equilibrium of a heavy
    inextensible chain. Returns None if s <= chord (no slack to hang).

    Worth preferring for the START shape: with a straight rest configuration and
    v0=0, any non-equilibrium start swings, and that transient pollutes the whole
    tracking horizon. A catenary is (up to bending stiffness, small here relative
    to gravity) already at rest, so the rope stays put until the arms move it."""
    d = B - A
    h = float(np.linalg.norm(d[:2]))          # horizontal span
    v = float(d[2])                           # rise
    if s <= np.hypot(h, v) * (1.0 + 1e-12) or h < 1e-9:
        return None
    rhs = np.sqrt(s * s - v * v)              # = 2a sinh(h/2a), decreasing in a
    lo, hi = 1e-9, 1.0
    while 2.0 * hi * np.sinh(h / (2.0 * hi)) > rhs:
        hi *= 2.0
        if hi > 1e6:
            return None
    for _ in range(200):                      # bisect for the catenary parameter a
        a = 0.5 * (lo + hi)
        if 2.0 * a * np.sinh(h / (2.0 * a)) > rhs:
            lo = a
        else:
            hi = a
    a = 0.5 * (lo + hi)
    mm = np.arctanh(np.clip(v / s, -1 + 1e-15, 1 - 1e-15))
    x0 = 0.5 * h - a * mm
    x = np.linspace(0.0, h, m)
    z = a * np.cosh((x - x0) / a)
    z += A[2] - z[0]
    ehat = np.array([d[0], d[1], 0.0]) / h
    return A[None, :] + x[:, None] * ehat[None, :] + (z - A[2])[:, None] * WORLD_UP[None, :]


def curve_between(shape, A, B, alpha, m=M_DENSE):
    """Dense curve from A to B. `alpha` scales transverse magnitude (bezier/helix)
    or the arc-length-to-chord ratio (catenary); it is what fit_length solves for."""
    R, d = chord_frame(A, B)
    if shape.kind == "catenary":
        P = _catenary(A, B, alpha * d, m)
        if P is None:
            return np.linspace(A, B, m)       # degenerate: fit_length rejects it
        return P
    if shape.kind == "helix":
        s = np.linspace(0.0, 1.0, m)
        a = 2.0 * np.pi * shape.turns * s
        r = alpha * shape.radius
        local = np.stack([s * d, r * (np.cos(a) - 1.0), r * np.sin(a)], axis=1)
        P = A + (R @ local.T).T
        # helix with non-integer turns does not land on B: blend the mismatch out
        P += np.linspace(0.0, 1.0, m)[:, None] * (B - P[-1])[None, :]
        return P
    ctrl = [A]
    for (x, y, z) in shape.ctrl:
        ctrl.append(A + d * (x * R[:, 0] + alpha * (y * R[:, 1] + z * R[:, 2])))
    ctrl.append(B)
    return bezier(ctrl, m)


def _closure(shape, A, B, alpha, e, k):
    """Signed slack after marching k chords of length e: >0 the curve is too long
    (polygon stops short of B), <0 it is too short to fit k chords at all.
    Monotone increasing in alpha, so bisection is safe."""
    P = curve_between(shape, A, B, alpha)
    V, i = equal_chord_polygon(P, e, k)
    if V is None:
        return -1.0
    return float(np.linalg.norm(V[-1] - P[i]) + arclen(P[i:]))


def fit_length(shape, A, B, L, segments, tol=1e-12, amax=60.0):
    """Solve the bulge scale alpha so that k=segments chords of length L/segments
    start at A and close exactly on B -- i.e. the POLYGON has length L and uniform
    edges, which is what the simulator's rest lengths actually are."""
    d = float(np.linalg.norm(B - A))
    if L <= d * (1.0 + 1e-9):
        raise ValueError(f"length {L:.4f} m <= chord {d:.4f} m: rope cannot span "
                         f"these endpoints without stretching")
    e = L / segments
    lo, hi = 0.0, 1.0
    while _closure(shape, A, B, hi, e, segments) < 0.0:
        hi *= 2.0
        if hi > amax:
            raise ValueError(f"shape cannot reach length {L:.4f} m (chord {d:.4f} m); "
                             f"deepen the control points or shorten the rope")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _closure(shape, A, B, mid, e, segments) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi        # the side where marching closes; lo is the failing side


def nominal_endpoints(shape, length, mid, rpy):
    """Where the rope's ends land for a given shape/length/placement, BEFORE IK.
    The chord is a consequence of shape and length (a saggier rope spans less),
    so it is derived here rather than being a free parameter."""
    A0, B0 = np.zeros(3), np.array([1.0, 0.0, 0.0])
    dense = curve_between(shape, A0, B0, shape.nominal_alpha())
    chord_frac = 1.0 / arclen(dense)          # chord / arc for this shape
    d = chord_frac * length
    axis = rpy_matrix(rpy) @ CHORD_AXIS
    mid = np.asarray(mid, float)
    return mid - 0.5 * d * axis, mid + 0.5 * d * axis


def build_rest(length, segments, A, B):
    """REST configuration: a STRAIGHT line of `segments` uniform edges.

    Rest shape and initial shape are different fields -- rest lengths/curvature come
    from the rope yaml `vertices:`, the initial state from the scenario `q0_rope`.
    Making rest straight is the standard elastic-rod convention: rest curvature is
    zero, so bending energy is proportional to actual curvature, and both the start
    sag and the goal shape carry honest stored bending. (Reusing the sagged start as
    rest instead would declare that sag unstressed, and the goal's bending energy
    would be measured against an arbitrary reference.)

    Placement is physically irrelevant -- rest lengths and rest curvature are
    invariant to rigid motion -- but it is laid along the start chord so it reads
    sensibly in any viewer."""
    e1 = (B - A) / np.linalg.norm(B - A)
    mid = 0.5 * (A + B)
    s = np.linspace(-0.5 * length, 0.5 * length, segments + 1)
    return mid[None, :] + s[:, None] * e1[None, :]


def build_rope(shape, A, B, length, segments):
    """Exact endpoints, exact length, uniform edges."""
    alpha = fit_length(shape, A, B, length, segments)
    P = curve_between(shape, A, B, alpha)
    V, _ = equal_chord_polygon(P, length / segments, segments)
    if V is None:
        raise ValueError("equal-chord marching failed after alpha fit")
    V[-1] = B                      # snap out the bisection residual
    V[0] = A
    return V


# ---------------------------------------------------------------- kinematics
def fk(model, data, q, fids):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return [np.array(data.oMf[f].translation) for f in fids]


def ik_two_frames(model, data, fids, q_init, targets, iters=400, damp=1e-5, tol=IK_TOL):
    """Damped least-squares IK driving both EE POSITIONS to `targets`."""
    q = q_init.copy()
    e = np.zeros(3 * len(fids))
    for _ in range(iters):
        ps = fk(model, data, q, fids)
        e = np.concatenate([targets[i] - ps[i] for i in range(len(fids))])
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((3 * len(fids), model.nv))
        pin.computeJointJacobians(model, data, q)
        for i, f in enumerate(fids):
            Jf = pin.getFrameJacobian(model, data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J[3 * i:3 * i + 3] = Jf[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(J.shape[0]), e)
        q = pin.integrate(model, q, dq)
    return q, float(np.linalg.norm(e))


def solve_placement(model, data, fids, q_seed, shape, length, segments, mid, rpy):
    """Place the rope, solve the arms to its ends, then REBUILD the rope between
    the achieved end-effector positions at the requested length. Returns
    (q, rope, ik_residual)."""
    A, B = nominal_endpoints(shape, length, mid, rpy)
    q, res = ik_two_frames(model, data, fids, q_seed, [A, B])
    Aa, Ba = fk(model, data, q, fids)          # achieved -> pin-exact by construction
    rope = build_rope(shape, Aa, Ba, length, segments)
    return q, rope, res


# ---------------------------------------------------------------- yaml writers
def _rows(V):
    return "\n".join(f"  - [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]" for v in V)


def write_rope_yaml(path, case, Vrest, pinned):
    with open(path, "w") as f:
        f.write(f"""# AUTO-GENERATED by gen_rope_cases.py -- case '{case.name}'
# length={case.length} m  segments={case.segments}  edge={case.length/case.segments:.6f} m
# This is the REST configuration: a STRAIGHT line, so rest curvature is zero and
# bending energy is proportional to actual curvature. It is NOT the initial state --
# that is q0_rope in the scenario yaml, and it is deliberately different (the start
# sag is bending-stressed against this straight reference, as a real rope is).
# NOTE: this assumes the loader uses `vertices:` only for rest lengths/curvature.
# If it also seeds the ADMM warm start or the pin positions, that has to come from
# q0_rope instead.
vertices:
{_rows(Vrest)}

mass:
  mass_uniform: true
  mass: {ROPE_MASS}
  mass_arr: []

damping:
  damping_uniform: true
  damping: {ROPE_DAMPING}
  damping_arr: []

stretching:
  enable: true
  stiffness: {ROPE_KS:g}

bending:
  enable_bending: false
  bending_as_force: true
  k_bend: {ROPE_KB:g}

control:
  pinned: [{pinned[0]}, {pinned[1]}]
""")


def write_scenario_yaml(path, case, rope_yaml, q0, qf, res0, resf, V0, Vf):
    z = "[" + ", ".join(["0.0"] * 14) + "]"
    s0 = "[" + ", ".join(f"{x:.6f}" for x in q0) + "]"
    sf = "[" + ", ".join(f"{x:.6f}" for x in qf) + "]"
    with open(path, "w") as f:
        f.write(f"""# AUTO-GENERATED by gen_rope_cases.py -- case '{case.name}'
# ROPE-FIRST: length={case.length} m, segments={case.segments}, shape start->goal.
# GOAL IS THE ROPE SHAPE (qf_rope). The terminal arm config is NOT pinned.
#   q0_arm: IK onto the start rope ends (residual {res0:.2e} m). Hard-pinned, and
#           the rope was rebuilt on the achieved ends, so it is exactly consistent.
#   qf_arm: GUESS ONLY (residual {resf:.2e} m) -- seeds the q0->qf interpolation in
#           buildInitialGuess. Nothing constrains the arm at the final node.
urdf: "{URDF}"
ropeYaml: "{rope_yaml}"
pinFrames: ["{FRAMES[0]}", "{FRAMES[1]}"]

deg: {DEG}
T: {T_HORIZON}
diffadmm_steps: {DIFFADMM_STEPS}
pfill: true
mu0: 10
mugrow: 3
tol: 1e-6
dsSeg: 5e-2
sMax: 2
muMax: 1e7

q0_arm: {s0}
qf_arm: {sf}
v0_arm: {z}
vf_arm: {z}

q0_rope:
{_rows(V0)}
v0_rope:
{chr(10).join(['  - [0, 0, 0]'] * len(V0))}

qf_rope_enable: true
qf_rope:
{_rows(Vf)}
""")


# ---------------------------------------------------------------- diagnostics
def report(case, V0, Vf, res0, resf):
    e0 = np.linalg.norm(np.diff(V0, axis=0), axis=1)
    ef = np.linalg.norm(np.diff(Vf, axis=0), axis=1)
    nominal = case.length / case.segments
    strain = float(np.abs(np.concatenate([e0, ef]) / nominal - 1.0).max())
    ch0 = float(np.linalg.norm(V0[-1] - V0[0]))
    chf = float(np.linalg.norm(Vf[-1] - Vf[0]))
    t1 = float(np.linalg.norm(Vf[0] - V0[0]))
    t2 = float(np.linalg.norm(Vf[-1] - V0[-1]))

    print(f"\n[{case.name}] L={case.length} m  seg={case.segments}  edge={nominal:.4f} m")
    print(f"  arc  {arclen(V0):.4f} -> {arclen(Vf):.4f} m   edge strain max {strain*100:.4f}%")
    print(f"  chord {ch0:.4f} -> {chf:.4f} m   (arc/chord {case.length/max(chf,1e-9):.3f})")
    print(f"  EE travel: arm1 {t1:.3f} m, arm2 {t2:.3f} m")
    print(f"  IK residual: q0 {res0:.2e} m   qf(guess) {resf:.2e} m")

    if strain > 1e-3:
        print(f"  WARN edges deviate {strain*100:.2f}% from nominal; at ks={ROPE_KS:g} "
              f"the stretch term will fight the tracking cost")
    if res0 > IK_WARN_RES:
        print(f"  WARN q0 IK did not converge -- start placement is out of reach; "
              f"move `mid` or shorten the rope")
    if resf > IK_WARN_RES:
        print(f"  WARN goal endpoints unreachable ({resf:.1e} m): the tracking cost "
              f"can never be satisfied at this goal placement")
    if min(ch0, chf) < MIN_EE_SEP:
        print(f"  WARN endpoints {min(ch0,chf):.3f} m apart (< {MIN_EE_SEP}) -- arms may collide")
    if max(t1, t2) > MAX_EE_TRAVEL:
        print(f"  WARN EE travel {max(t1,t2):.3f} m in T={T_HORIZON}s may be infeasible")
    zmin = min(V0[:, 2].min(), Vf[:, 2].min())
    if zmin < FLOOR_Z:
        print(f"  WARN rope dips to z={zmin:.3f} (below floor {FLOOR_Z})")


def preview(case, V0, Vf, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return False
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*V0.T, "-o", ms=3, lw=2, color="tab:blue", label="start")
    ax.plot(*Vf.T, "--o", ms=3, lw=2, color="tab:orange", label="goal")
    ax.scatter(*V0[[0, -1]].T, color="red", s=40, zorder=5)
    P = np.vstack([V0, Vf])
    c, r = P.mean(0), 0.6 * np.ptp(P, axis=0).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_title(case.name); ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return True


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_preview = "--preview" in sys.argv
    wanted = set(args)
    cases = [c for c in CASES if not wanted or c.name in wanted]
    if not cases:
        sys.exit(f"no cases matched {sorted(wanted)}; known: {[c.name for c in CASES]}")

    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    fids = [model.getFrameId(n) for n in FRAMES]
    assert model.nq == 14, f"expected 14-DOF 2-arm model, got nq={model.nq}"
    q_seed = np.array(Q_SEED, float)

    if do_preview:
        import os
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    print(f"[cfg] deg={DEG} (N_n={DEG+1})  ks={ROPE_KS:g}  kb={ROPE_KB:g}  "
          f"dt={T_HORIZON/(DIFFADMM_STEPS-1):.4f}  T={T_HORIZON}")

    for case in cases:
        try:
            q0, V0, res0 = solve_placement(model, data, fids, q_seed, case.start,
                                           case.length, case.segments, case.mid, case.rpy)
            gmid, grpy = case.goal_placement()
            # seed the goal IK from q0: the arms should not teleport between them
            qf, Vf, resf = solve_placement(model, data, fids, q0, case.goal,
                                           case.length, case.segments, gmid, grpy)
        except ValueError as err:
            print(f"\n[{case.name}] SKIP: {err}")
            continue

        report(case, V0, Vf, res0, resf)

        rope_path = f"{ROPE_DIR}/arm_rope_{case.name}.yaml"
        scen_path = f"{SCEN_DIR}/2arm1rope_{case.name}.yaml"
        rope_dir = Path(ROPE_DIR)
        scen_dir = Path(SCEN_DIR)
        rope_dir.mkdir(parents=True, exist_ok=True)
        scen_dir.mkdir(parents=True, exist_ok=True)
        Vrest = build_rest(case.length, case.segments, V0[0], V0[-1])
        write_rope_yaml(rope_path, case, Vrest, [0, case.segments])
        write_scenario_yaml(scen_path, case, rope_path, q0, qf, res0, resf, V0, Vf)
        print(f"  write {rope_path}\n  write {scen_path}")

        if do_preview:
            p = f"{PREVIEW_DIR}/{case.name}.png"
            print(f"  write {p}" if preview(case, V0, Vf, p) else "  (matplotlib absent)")

    print("\nRun:  ./build/rope_admm alm backend=eigen  (point it at a 2arm1rope_*.yaml)")