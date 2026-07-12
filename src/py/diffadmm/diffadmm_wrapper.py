from dataclasses import dataclass, field
from typing import List
from yaml import safe_load
import numpy as np

import src.py.diffadmm.diffadmm as diffadmm


# TODO: support nonzero velocity ic
@dataclass
class RopeProperties:

    N: int = 0
    n_u: int = 0

    x0: np.ndarray = None  # (N, 3)
    v0: np.ndarray = None  # (N, 3)
    l0: np.ndarray = None  # N-1
    mass: np.ndarray = None  # N
    damping: np.ndarray = None  # N

    pinned: np.ndarray = None

    enable_stretching: bool = False
    enable_bending: bool = False
    bending_as_force: bool = False

    k_stretch: np.ndarray = None  # N-1
    k_bend: np.ndarray = None  # N-2

    @staticmethod
    def from_yaml(path):

        with open(path, "r") as f:
            data = safe_load(f)

        # Vertices
        r = RopeProperties()
        r.x0 = np.array(data["vertices"])
        r.N = r.x0.shape[0]
        r.v0 = np.zeros((r.N, 3))

        # L0, currently calculated from resting position. maybe change
        r.l0 = np.linalg.norm(r.x0[1:] - r.x0[:-1], axis=1)

        # Mass
        if data["mass"]["mass_uniform"]:
            r.mass = np.full(shape=(r.B, r.N), fill_value=data["mass"]["mass"])
        else:
            if len(data["mass"]["mass_arr"]) != r.N:
                raise ValueError("Wrong mass size")
            r.mass = np.array(data["mass"]["mass_arr"])

        # Damping
        if data["damping"]["damping_uniform"]:
            r.damping = np.full(shape=r.N, fill_value=data["damping"]["damping"])
        else:
            damping_arr = np.array(data["damping"]["damping_arr"])
            if len(damping_arr) != r.N:
                raise ValueError(f"Wrong damping size")
            r.damping = damping_arr

        # Stretching
        r.enable_stretching = data["stretching"]["enable"]
        r.k_stretch = np.full(
            shape=max(r.N - 1, 0),
            fill_value=data["stretching"]["stiffness"] if r.enable_stretching else 0.0,
        )

        # Bending
        r.enable_bending = data["bending"]["enable_bending"]
        r.bending_as_force = data["bending"]["bending_as_force"]

        bending = r.enable_bending or r.bending_as_force
        r.k_bend = np.full(
            shape=max(r.N - 2, 0),
            fill_value=data["bending"]["k_bend"] if bending else 0.0,
        )

        # Control (pinned)
        r.pinned = data["control"]["pinned"]
        r.n_u = len(r.pinned) * 3

        return r


@dataclass
class DeformableConfig:
    source: str

    dt: float = 0.05
    T: int = 20
    DEFAULT_BATCH_SIZE: int = 1 # Keep in case Jxu scratch ever is exposed
    gmres_m: int = 150
    gmres_restart: int = 60
    gmres_tol: float = 1e-8
    ADMM_Iter: int = 500

    stretch_pen: float = 1e3
    bend_pen: float = 0.0


class Deformable:
    """
    Wraps the diffadmm methods. Useful utilities.
    """

    # TODO modify pybind wrapper to expose A_inv
    # so that can be precomputed
    def __init__(self, cfg: DeformableConfig):
        self.config = cfg
        self.r = RopeProperties.from_yaml(self.config.source)

    def _batched(self, B):
        N = self.r.N

        def rep(a):
            # Repeats a B times in the first dimension
            return np.repeat(np.asarray(a, np.float64)[None, ...], B, axis=0)

        return dict(
            x0=rep(self.r.x0.reshape(-1)),  # (B, 3N)
            v0=rep(self.r.v0.reshape(-1)),  # (B, 3N)
            mass=rep(self.r.mass),  # (B, N)
            L0=rep(self.r.l0),  # (B, N-1)
            damping=rep(self.r.damping),  # (B, N)
            stiffness_stretch=rep(self.r.k_stretch),  # (B, N-1)
            stiffness_bend=rep(self.r.k_bend),  # (B, N-2)
            penalty_stretch=np.full((B, N - 1), self.config.stretch_pen, np.float64),
        )

    def _flags(self):
        return dict(
            bending_admm=self.r.enable_bending,
            stretching_admm=self.r.enable_stretching,
            bending_as_force=self.r.bending_as_force,
        )

    def forwards(self, pin_pos):
        # pin_pos: (B, T, n_pins, 3), n_pins == len(pinned), T == config.T
        pin_pos = np.ascontiguousarray(pin_pos, dtype=np.float64)
        B = pin_pos.shape[0]

        return diffadmm.forward(
            **self._batched(B),
            pin_indices=self.r.pinned,
            pin_positions=pin_pos,
            dt=self.config.dt,
            penalty_bend=self.config.bend_pen,
            ADMM_ITERS=self.config.ADMM_Iter,
            T=self.config.T,
            admm_tol=0.0,
            admm_check_interval=0,
            **self._flags(),
        )

    def jac(self, pin_pos, t):
        # pin_pos: (B, T, n_pins, 3)
        # t must be an array/list
        # Returns in shape (t, B, N3, n_pins)

        pin_pos = np.ascontiguousarray(pin_pos, dtype=np.float64)
        B = pin_pos.shape[0]

        Jxu, _, _  = diffadmm.forwards_with_jac(
            **self._batched(B),
            pin_indices=self.r.pinned,
            pin_positions=pin_pos,
            dt=self.config.dt,
            penalty_bend=self.config.bend_pen,
            ADMM_ITERS=self.config.ADMM_Iter,
            T=self.config.T,
            admm_tol=0.0,
            admm_check_interval=0,
            t=t,
            gmres_m=self.config.gmres_m,
            gmres_restart=self.config.gmres_restart,
            gmres_tol=self.config.gmres_tol,
            **self._flags(),
        )

        return Jxu.reshape(len(t), B, 3 * self.r.N, self.r.n_u)
