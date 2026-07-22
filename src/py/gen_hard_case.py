"""
gen_hard_case.py — build a HARDER rope-manipulation test:
  * arms start ASYMMETRIC (arm1 != arm2), and
  * the goal config COMPRESSES the end-to-end span so the rope buckles
    (bending_as_force resists -> stored bending during the motion).
 
Consistency is guaranteed by construction:
  * rope pin vertices v0, v_{N-1} == FK(arm end-effectors) at q0, so the cpp
    "pin vertex ... from frame" warning cannot fire.
  * qf is solved by IK to hit a target end-effector span = COMPRESS * span(q0),
    keeping the midpoint fixed, so the compression is achieved regardless of
    the arm kinematics (no hand-tuning).
  * rope initial shape is a half-sine arch between the two pins (matches the
    original rest-curvature convention). Since rest length/curvature are derived
    from x0, the arch IS the rest shape; the *compression* (hence bending force)
    comes from qf pulling the ends in during the trajectory.
 
Writes two files matching the existing schema:
  arm_rope_hard.yaml       (rope: vertices/mass/damping/stretching/bending/control)
  2arm1rope_hard.yaml      (scenario: urdf/pins/deg/T/q0_arm/qf_arm/q0_rope/...)
 
Run from repo root (needs pinocchio + the URDF).
"""
 
import numpy as np
import pinocchio as pin
 
# ---------------------------------------------------------------- config
URDF   = "./src/data/urdf/kinova3_2arm.urdf"
FRAMES = ["arm1_end_effector_link", "arm2_end_effector_link"]
N      = 10                     # rope vertices
SAG    = 0.09                   # arch amplitude [m] (rest curvature; larger => stiffer arch)
COMPRESS = 0.72                 # target span(qf) / span(q0)   (<1 compresses the rope)
T_HORIZON = 2.0
DIFFADMM_STEPS = 21
 
# asymmetric start: arm1 and arm2 in DIFFERENT configs (7 DOF each)
Q0_ARM1 = [ 0.35, -0.45,  0.25,  0.30, 0.0, 0.0, 0.0]
Q0_ARM2 = [-0.30,  0.55, -0.20, -0.35, 0.0, 0.0, 0.0]
 
ROPE_OUT = "./src/data/rope/arm_rope_hard.yaml"
SCEN_OUT = "./src/data/problems/2arm1rope_hard.yaml"
 
 
# ---------------------------------------------------------------- FK / IK
def fk(model, data, q, fids):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return [np.array(data.oMf[f].translation) for f in fids]
 
 
def ik_two_frames(model, data, fids, q_init, targets, iters=200, damp=1e-4, tol=1e-8):
    """Damped least-squares IK driving two frame POSITIONS to `targets`."""
    q = q_init.copy()
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
        # GN step with Levenberg damping
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(J.shape[0]), e)
        q = pin.integrate(model, q, dq)
    return q, np.linalg.norm(e)
 
 
# ---------------------------------------------------------------- rope shape
def build_arch(pa, pb, n, sag):
    """n vertices from pa to pb along a half-sine arch bowing 'up' (max +z)."""
    chord = pb - pa
    L = np.linalg.norm(chord)
    chat = chord / L
    up = np.array([0.0, 0.0, 1.0])
    perp = up - (up @ chat) * chat            # component of world-up perp to chord
    if np.linalg.norm(perp) < 1e-6:           # chord ~vertical: pick any perp
        perp = np.cross(chat, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    s = np.linspace(0.0, 1.0, n)
    V = pa[None, :] + s[:, None] * chord[None, :] + (sag * np.sin(np.pi * s))[:, None] * perp[None, :]
    return V                                   # (n,3), V[0]=pa, V[-1]=pb exactly
 
 
# ---------------------------------------------------------------- yaml writers
def _fmt_rows(V):
    return "\n".join(f"  - [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]" for v in V)
 
 
def write_rope_yaml(path, V, pinned):
    txt = f"""# AUTO-GENERATED harder case (gen_hard_case.py)
# N={len(V)} arch sag={SAG} m ; pins {pinned} == FK(end-effectors) at q0
vertices:
{_fmt_rows(V)}
 
mass:
  mass_uniform: true
  mass: 0.2
  mass_arr: []
 
damping:
  damping_uniform: true
  damping: 0.1
  damping_arr: []
 
stretching:
  enable: true
  stiffness: 1e6
 
bending:
  enable_bending: false
  bending_as_force: true
  k_bend: 1e3
 
control:
  pinned: [{pinned[0]}, {pinned[1]}]
"""
    with open(path, "w") as f:
        f.write(txt)
 
 
def write_scenario_yaml(path, rope_yaml, q0_arm, qf_arm, V):
    z = "[" + ", ".join(["0.0"] * 14) + "]"
    q0 = "[" + ", ".join(f"{x:.4f}" for x in q0_arm) + "]"
    qf = "[" + ", ".join(f"{x:.4f}" for x in qf_arm) + "]"
    txt = f"""# AUTO-GENERATED harder case (gen_hard_case.py)
urdf: "{URDF}"
ropeYaml: "{rope_yaml}"
pinFrames: ["{FRAMES[0]}", "{FRAMES[1]}"]
 
deg: 16
T: {T_HORIZON}
diffadmm_steps: {DIFFADMM_STEPS}
pfill: true
mu0: 10
mugrow: 3
tol: 1e-6
dsSeg: 5e-2
sMax: 2
muMax: 1e7
 
q0_arm: {q0}
qf_arm: {qf}
v0_arm: {z}
vf_arm: {z}
 
q0_rope:
{_fmt_rows(V)}
v0_rope:
{chr(10).join(['  - [0, 0, 0]'] * len(V))}
 
qf_rope_enable: false
qf_rope: []
"""
    with open(path, "w") as f:
        f.write(txt)
 
 
# ---------------------------------------------------------------- main
if __name__ == "__main__":
    model = pin.buildModelFromUrdf(URDF)
    data = model.createData()
    fids = [model.getFrameId(n) for n in FRAMES]
    assert model.nq == 14, f"expected 14-DOF 2-arm model, got nq={model.nq}"
 
    q0 = np.array(Q0_ARM1 + Q0_ARM2, float)
    pa, pb = fk(model, data, q0, fids)
    span0 = np.linalg.norm(pb - pa)
    print(f"[q0] span(arm1_EE, arm2_EE) = {span0:.4f} m  asymmetric start")
 
    # target: same midpoint, compressed span
    mid = 0.5 * (pa + pb)
    dirv = (pb - pa) / span0
    half = 0.5 * COMPRESS * span0
    ta, tb = mid - half * dirv, mid + half * dirv
    qf, err = ik_two_frames(model, data, fids, q0, [ta, tb])
    pa_f, pb_f = fk(model, data, qf, fids)
    span_f = np.linalg.norm(pb_f - pa_f)
    print(f"[qf] IK residual = {err:.2e} m ; span(qf) = {span_f:.4f} m "
          f"=> compression {span_f/span0:.3f} (target {COMPRESS})")
    if err > 1e-3:
        print("  WARNING: IK did not converge; loosen COMPRESS or adjust Q0_ARM*")
 
    # rope arch between the ACTUAL q0 end-effector positions (pins consistent)
    V = build_arch(pa, pb, N, SAG)
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    print(f"[rope] rest edges {seg.min():.4f}..{seg.max():.4f} m ; "
          f"arc length {seg.sum():.4f} m vs chord {span0:.4f} m")
    print(f"[rope] compressed chord at qf {span_f:.4f} m => "
          f"arc/chord {seg.sum()/span_f:.3f} (>1 buckles)")
 
    pinned = [0, N - 1]
    write_rope_yaml(ROPE_OUT, V, pinned)
    write_scenario_yaml(SCEN_OUT, ROPE_OUT, q0, qf, V)
    print(f"[write] {ROPE_OUT}\n[write] {SCEN_OUT}")
    print("Run:  ./build/rope_admm alm backend=eigen  (point it at 2arm1rope_hard.yaml)")