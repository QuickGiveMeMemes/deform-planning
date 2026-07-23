import numpy as np
import pinocchio as pin

"""
Plotting util functions. Can probably do something nicer with actual 3d models
"""


# Pinocchio plot utils

def x_at_config(model, data, q):
    """
    Forward Kinematics plotting utility
    """

    pin.forwardKinematics(model, data, q)
    x = np.zeros((model.njoints, 3))
    for i, tup in enumerate(zip(model.names, data.oMi)):
        name, oMi = tup
        x[i, :] = oMi.translation.T
    return x

def gen_plt_from_urdf(ax, model, data, q=None, pt_kwargs=None, seg_kwargs=None):
    """
    Initializes segments corresponding to robot
    """
    pt_kwargs = {} if pt_kwargs is None else pt_kwargs
    seg_kwargs = {} if seg_kwargs is None else seg_kwargs

    if q is None:
        q = np.zeros(model.njoints - 1)

    x0 = x_at_config(model, data, q)

    segs = []
    for i in range(1, model.njoints):
        if model.parents[i] == 0:
            continue

        j_idx = i
        p_idx = model.parents[i]
        x = np.vstack(
            (
                x0[j_idx, :],
                x0[p_idx, :],
            )
        )
        seg, = ax.plot3D(x[:, 0], x[:, 1], x[:, 2], **seg_kwargs)
        segs.append((j_idx, p_idx, seg))

    ax.set_axis_on()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return segs

def upd_plt_arm(segs, model, data, q):
    """
    Updates segments to config
    """

    x0 = x_at_config(model, data, q)

    for i, seg in enumerate(segs):
        j_idx, p_idx, seg_p = seg

        seg_p.set_data_3d(
            np.vstack(
                (
                    x0[j_idx, :],
                    x0[p_idx, :],
                )
            ).T
        )



def plot_init_rope(ax, x: np.ndarray):
    # Assuming dims (T, 3N) so batch already removed
    if len(x.shape) == 2:
        T = x.shape[0]
        N = x.shape[1] // 3

        x = x.reshape((T, N, 3))
    else:
        T = x.shape[0]
        N = x.shape[1]

    segs = []
    for _ in range(1, N):
        (seg,) = ax.plot3D(x[:, 0], x[:, 1], x[:, 2])
        segs.append(seg)

    ax.set_axis_on()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return segs

def upd_plt_rope(segs, x, t):

    if len(x.shape) == 2:
        T = x.shape[0]
        N = x.shape[1] // 3

        x = x.reshape((T, N, 3))
    else:
        T = x.shape[0]
        N = x.shape[1]


    for i, seg in enumerate(segs):
        seg.set_data_3d(
            np.vstack(
                (
                    x[t, i, :],
                    x[t, i + 1, :],
                )
            ).T
        )
