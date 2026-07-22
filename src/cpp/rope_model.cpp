#include "../../include/rope_model.hpp"
// #include "admm_backward.h"
#include "admm_forward.h"
#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <vector>

DynamicRopeModel::DynamicRopeModel(const RopeParams r) : r(std::move(r)) {

    const int N = r.rest_pos.rows();

    rest_length = Eigen::VectorXd(N - 1);
    rest_curv = Eigen::VectorXd(N - 2);

    for (int i = 0; i < N - 1; ++i) {
        // Assumes initial configuration is rest length
        rest_length(i) = (r.rest_pos.row(i + 1) - r.rest_pos.row(i)).norm();
    }

    for (int i = 0; i < N - 2; ++i) {
        // Assumes initial configuration is rest curvature
        rest_curv(i) =
            (r.rest_pos.row(i) - 2 * r.rest_pos.row(i + 1) + r.rest_pos.row(i + 2)).norm();
    }
}

std::tuple<std::vector<int>, std::vector<int>> DynamicRopeModel::split() const {

    const int N = r.rest_pos.rows();

    // Mask for efficiency
    std::vector<bool> pinned(N, false);
    for (auto i : r.pinned_idx)
        pinned[i] = true;

    std::vector<int> p, f;
    p.reserve(r.pinned_idx.size());
    f.reserve(N - r.pinned_idx.size());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            (pinned[i] ? p : f).push_back(3 * i + j);
        }
    }

    return {f, p};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd>
DynamicRopeModel::elastic_grad(const Eigen::Ref<const Eigen::MatrixXd> &R, int mode) const {

    const bool bend = mode == 1;
    const int N = this->r.rest_pos.rows();

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(N * 3);

    for (int i = (bend ? 1 : 0); i < N - 1; ++i) {

        // Resting curvature/length, stiffness const
        double rest, k;
        Eigen::Vector3d g;

        // Stuff for stencil (last not used for stretch)
        std::array<int, 3> idx;
        std::array<double, 3> stencil;

        int l;

        if (bend) {
            l = 3;
            rest = rest_curv(i - 1);
            k = this->r.kbend(i - 1);
            g = (R.row(i - 1) - 2 * R.row(i) + R.row(i + 1)).transpose();
            idx = {i - 1, i, i + 1};
            stencil = {1, -2, 1};
        } else {
            l = 2;
            rest = rest_length(i);
            k = this->r.kstretch(i);
            g = (R.row(i + 1) - R.row(i)).transpose();
            idx = {i, i + 1, 0};
            stencil = {-1, 1, 0};
        }

        const double norm = std::max(g.norm(), 1e-9);
        const Eigen::Vector3d ghat = g / norm;
        const Eigen::Vector3d f =
            k * (norm - rest) * ghat; // Term k( ||g|| - rest ) ghat (before D^T matmul)

        for (int a = 0; a < l; ++a)
            grad.segment(3 * idx[a], 3) += stencil[a] * f;
    }

    const auto [free, pinned] = split();
    return {grad(free), grad(pinned)};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
DynamicRopeModel::elastic_hess(const Eigen::Ref<const Eigen::MatrixXd> &R, int mode) const {

    const bool bend = mode == 1;
    const int N = this->r.rest_pos.rows();

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(N * 3, N * 3);

    for (int i = (bend ? 1 : 0); i < N - 1; ++i) {

        // Resting curvature/length, stiffness const
        double rest, k;
        Eigen::Vector3d g;

        // Stuff for stencil (last not used for stretch)
        std::array<int, 3> idx;
        std::array<double, 3> stencil;

        int l;

        if (bend) {
            l = 3;
            rest = rest_curv(i - 1);
            k = this->r.kbend(i - 1);
            g = (R.row(i - 1) - 2 * R.row(i) + R.row(i + 1)).transpose();
            idx = {i - 1, i, i + 1};
            stencil = {1, -2, 1};
        } else {
            l = 2;
            rest = rest_length(i);
            k = this->r.kstretch(i);
            g = (R.row(i + 1) - R.row(i)).transpose();
            idx = {i, i + 1, 0};
            stencil = {-1, 1, 0};
        }

        const double norm = std::max(g.norm(), 1e-9);
        const Eigen::Vector3d ghat = g / norm;

        const Eigen::Matrix3d gg = ghat * ghat.transpose();
        const Eigen::Matrix3d A = k * (gg + (1 - rest / norm) * (Eigen::Matrix3d::Identity() - gg));

        for (int a = 0; a < l; ++a)
            for (int b = 0; b < l; ++b) {
                K.block<3, 3>(3 * idx[a], 3 * idx[b]) += stencil[a] * stencil[b] * A;
            }
    }

    const auto [free, pinned] = split();
    return {K(free, free), K(free, pinned), K(pinned, pinned)};
}

// Some helpers
namespace {
    using u_mat = Kokkos::View<double **, Kokkos::LayoutStride, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using u_vec = Kokkos::View<double *, Kokkos::LayoutStride, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    void build_kokkos_2d(Eigen::Ref<Eigen::MatrixXd> e, Kokkos::View<double **> &k_d,
                         std::string n) {
        Kokkos::LayoutStride stride( // So LayoutLeft/LayoutRight doesn't screw up
            e.rows(), e.innerStride(), e.cols(), e.outerStride());
        k_d = Kokkos::View<double **>(n, e.rows(), e.cols());
        auto k_h = Kokkos::create_mirror_view(k_d);
        u_mat k_unmanaged(e.data(), stride);
        Kokkos::deep_copy(k_h, k_unmanaged);
        Kokkos::deep_copy(k_d, k_h);
    }
    void build_kokkos_1d(Eigen::Ref<Eigen::VectorXd> e, Kokkos::View<double *> &k_d,
                         std::string n) {
        k_d = Kokkos::View<double *>(n, e.size());
        auto k_h = Kokkos::create_mirror_view(k_d);
        Kokkos::LayoutStride stride(e.size(), e.innerStride());
        u_vec k_unmanaged(e.data(), stride);
        Kokkos::deep_copy(k_h, k_unmanaged);
        Kokkos::deep_copy(k_d, k_h);
    }
    // (N) -> (1, N) for fake batch
    void build_kokkos_1d_batch(Eigen::Ref<Eigen::VectorXd> e, Kokkos::View<double **> &k_d,
                               const std::string &n) {
        const int len = e.size();
        k_d = Kokkos::View<double **>(n, 1, len);
        auto k_h = Kokkos::create_mirror_view(k_d);
        Kokkos::LayoutStride stride(len, e.innerStride());
        u_vec k_unmanaged(e.data(), stride);
        auto row0 = Kokkos::subview(k_h, 0, Kokkos::ALL);
        Kokkos::deep_copy(row0, k_unmanaged);
        Kokkos::deep_copy(k_d, k_h);
    }
    void build_eigen_2d(Kokkos::View<double **> k_d, Eigen::MatrixXd &e) {
        auto k_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_d);
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> map(
            k_h.data(), k_h.extent(0), k_h.extent(1),
            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(k_h.stride_1(), k_h.stride_0()));
        e = map;
    }
    void build_eigen_1d(Kokkos::View<double *> k_d, Eigen::VectorXd &e) {
        auto k_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k_d);
        Eigen::Map<Eigen::VectorXd> map(k_h.data(), k_h.extent(0));
        e = map;
    }
} // namespace

// Contains a lot of cache/scratch buffer creation + precomputation of a good few
// DiffADMM parameters
// TODO if batch size is ever needed everything is currently hardcoded to 1
// TODO should implement explicit rest_curv passthrough
ADMMRopeModel::ADMMRopeModel(ADMMParams _r) {
    r = _ADMMParams();
    const int N = _r.rest_pos.rows() / 3;

    // r.rest_pos_d = Kokkos::View<double **>("rest_pos", 1, 3 * N);

    // Copying over
    build_kokkos_1d_batch(_r.rest_pos, r.rest_pos_d, "rest_pos");
    build_kokkos_1d_batch(_r.kstretch, r.kstretch_d, "kstretch");
    build_kokkos_1d_batch(_r.kbend, r.kbend_d, "kbend");
    r.pinned_idx = std::move(_r.pinned_idx);

    const int npin = r.pinned_idx.size();

    Eigen::VectorXd mass_pv(N), damp_pv(N);
    for (int i = 0; i < N; ++i) {
        mass_pv(i) = _r.mass(3 * i);
        damp_pv(i) = _r.damping(3 * i);
    }
    build_kokkos_1d_batch(mass_pv, r.mass_d, "mass");       // (1, N)
    build_kokkos_1d_batch(damp_pv, r.damping_d, "damping"); // (1, N)

    r.dt = _r.dt;
    r.T = _r.T;
    r.DEFAULT_BATCH_SIZE = _r.DEFAULT_BATCH_SIZE;
    r.gmres_m = _r.gmres_m;
    r.gmres_restart = _r.gmres_restart;
    r.admm_iter = _r.admm_iter;
    r.gmres_tol = _r.gmres_tol;

    r.stretch_pen = _r.stretch_pen;
    r.bend_pen = _r.bend_pen;
    r.stretch_pen_d = Kokkos::View<double **>("stretch_pen", 1, N - 1);
    // r.bend_pen_d = Kokkos::View<double **>("bend_pen", 1, N - 2);
    {
        auto stretch_pen_h = Kokkos::create_mirror_view(r.stretch_pen_d);
        // auto bend_pen_h = Kokkos::create_mirror_view(r.bend_pen_d);
        for (int i = 0; i < N - 1; ++i)
            stretch_pen_h(0, i) = r.stretch_pen;
        // for (int i = 0; i < N - 2; ++i)
        //     bend_pen_h(0, i) = r.bend_pen;
        Kokkos::deep_copy(r.stretch_pen_d, stretch_pen_h);
        // Kokkos::deep_copy(r.bend_pen_d, bend_pen_h);
    }

    r.is_pinned_d = Kokkos::View<bool *>("is_pinned", N);
    {
        auto host = Kokkos::create_mirror_view(r.is_pinned_d);
        for (int i = 0; i < N; ++i)
            host(i) = false;
        for (int p : r.pinned_idx)
            host(p) = true;
        Kokkos::deep_copy(r.is_pinned_d, host);
    }
    A_inv_d = build_A_inv(r.mass_d, 1.0 / (r.dt * r.dt), r.stretch_pen_d, r.is_pinned_d, r.bend_pen,
                          false, true, true, r.kbend_d);
    b = allocate_backward_scratch(1, N, r.gmres_m, true, false, true);
    penaltyDt_stretch_d = build_penaltyDt_stretch_batch(1, N, N - 1, r.stretch_pen_d);
    D_bend_d = build_D_bend_batch(1, N);
    W_bend_d = build_W_bend_batch(1, N, r.bend_pen);

    // Rest len
    Eigen::VectorXd L0(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        // Assumes initial configuration is rest length
        L0(i) = (_r.rest_pos.segment((i + 1) * 3, 3) - _r.rest_pos.segment(i * 3, 3)).norm();
    }
    build_kokkos_1d_batch(L0, r.L0_d, "L0");

    // FwdOut size alloc
    fo.x_hist_d = Kokkos::View<double ***>("xhist", 1, r.T, 3 * N);
    fo.y_hist_d = Kokkos::View<double ***>("yhist", 1, r.T, 3 * N);
    fo.z_hist_d = Kokkos::View<double ***>("zhist", 1, r.T, 3 * (N - 1));
    fo.dual_hist_d = Kokkos::View<double ***>("dualhist", 1, r.T, 3 * (N - 1));
    fo.z_bend_hist_d = Kokkos::View<double ***>("zbend", 1, r.T, 3 * (N - 2));
    fo.dual_bend_hist_d = Kokkos::View<double ***>("dualbend", 1, r.T, 3 * (N - 2));

    // BwdOut size alloc
    bo.grad_pin_pos_d = Kokkos::View<double ****>("grad_pin_pos", 1, r.T, N, 3);
    // bo.grad_damping_d = Kokkos::View<double**>("grad_damping", 1, N);
    // bo.grad_stiffness_bend_d = Kokkos::View<double**>("grad_kbend", 1, N-2);
    // bo.grad_mass_d = Kokkos::View<double**>("grad_mass", 1, N);
}

void ADMMRopeModel::forward(Eigen::VectorXd x0, Eigen::VectorXd v0, Eigen::MatrixXd pin_pos) {
    const int npin = r.pinned_idx.size();
    const int N = r.rest_pos_d.extent(1) / 3;
    // Manual pinpos parsing (B,T,N,3)
    Kokkos::View<double ****> pin_pos_d("pin_pos", 1, r.T, N, 3);
    auto pin_pos_h = Kokkos::create_mirror_view(pin_pos_d);
    Kokkos::deep_copy(pin_pos_h, 0.0);
    for (int i = 0; i < r.T; ++i)
        for (int j = 0; j < npin; ++j)
            for (int k = 0; k < 3; ++k)
                pin_pos_h(0, i, r.pinned_idx[j], k) = pin_pos(i, j * 3 + k);
    Kokkos::deep_copy(pin_pos_d, pin_pos_h);

    Kokkos::View<double **> x0_d, v0_d;
    build_kokkos_1d_batch(x0, x0_d, "x0");
    build_kokkos_1d_batch(v0, v0_d, "v0");

    admm_forward(x0_d, v0_d, r.mass_d, r.L0_d, r.kstretch_d, r.stretch_pen_d, pin_pos_d,
                 r.is_pinned_d, A_inv_d, r.dt, 1 / (r.dt * r.dt), r.damping_d, r.admm_iter, r.T,
                 fo.x_hist_d, fo.y_hist_d, fo.z_hist_d, fo.dual_hist_d, fo.z_bend_hist_d,
                 fo.dual_bend_hist_d, r.kbend_d, r.bend_pen, false, true, true);
    fo.pin_pos_d = pin_pos_d;
    fo.x0_d = x0_d;
    fo.v0_d = v0_d;
}

void ADMMRopeModel::backward(Eigen::MatrixXd loss_per_grad_step) {
    const int N = r.rest_pos_d.extent(1) / 3;
    const int npin = static_cast<int>(r.pinned_idx.size());
    const int T = r.T;
    Kokkos::deep_copy(bo.grad_pin_pos_d, 0.0);
    // DEBUG generated a bunch of assert guards
    // ---- scalars: must be sane or the solver hangs / poisons A_inv ----
    assert(N >= 3 && "N < 3");
    assert(T > 0 && "r.T unset/garbage");
    assert(npin >= 1 && "no pinned vertices");
    assert(r.dt > 0.0 && "r.dt unset -> 1/dt^2 = inf");
    assert(r.admm_iter > 0 && "admm_iter unset");
    assert(r.gmres_m > 0 && "gmres_m unset");
    assert(r.gmres_restart > 0 && "gmres_restart unset");
    assert(r.gmres_tol > 0.0 && std::isfinite(r.gmres_tol) &&
           "gmres_tol unset/NaN -> no convergence");
    assert(std::isfinite(r.bend_pen) && "bend_pen unset");

    // ---- host input (Eigen) matches the flat state width ----
    assert(loss_per_grad_step.rows() == T && "loss seed row count != T");
    assert(loss_per_grad_step.cols() == 3 * N && "loss seed col count != 3N");

    // ---- forward histories: (B=1, T, field-width) ----
    auto chk3 = [](const auto &V, int d1, int d2, const char *nm) {
        assert(V.extent(0) == 1 && nm);
        (void)nm;
        assert(static_cast<int>(V.extent(1)) == d1 && nm);
        assert(static_cast<int>(V.extent(2)) == d2 && nm);
    };
    chk3(fo.x_hist_d, T, 3 * N, "x_hist");
    chk3(fo.y_hist_d, T, 3 * N, "y_hist");
    chk3(fo.z_hist_d, T, 3 * (N - 1), "z_hist");
    chk3(fo.dual_hist_d, T, 3 * (N - 1), "dual_hist");
    chk3(fo.z_bend_hist_d, T, 3 * (N - 2), "z_bend_hist");
    chk3(fo.dual_bend_hist_d, T, 3 * (N - 2), "dual_bend_hist");

    // ---- per-DOF / per-edge params (B=1, width) ----
    auto chk2 = [](const auto &V, int d1, const char *nm) {
        assert(V.extent(0) == 1 && nm);
        (void)nm;
        assert(static_cast<int>(V.extent(1)) == d1 && nm);
    };
    chk2(r.mass_d, N, "mass_d");                   // per-DOF
    chk2(r.damping_d, N, "damping_d");             // per-DOF  <-- the empty-View crash source
    chk2(r.L0_d, N - 1, "L0_d");                   // per-edge
    chk2(r.kstretch_d, N - 1, "kstretch_d");       // per-edge
    chk2(r.stretch_pen_d, N - 1, "stretch_pen_d"); // per-edge
    chk2(r.kbend_d, N - 2, "kbend_d");             // per-interior
    chk2(fo.x0_d, 3 * N, "x0_d");
    chk2(fo.v0_d, 3 * N, "v0_d");

    // ---- is_pinned: rank-1 (N,), bool ----
    assert(static_cast<int>(r.is_pinned_d.extent(0)) == N && "is_pinned_d width != N");

    // ---- pin trajectories: (B=1, T, npin, 3) ----
    auto chk4 = [&](const auto &V, const char *nm) {
        assert(V.extent(0) == 1 && nm);
        (void)nm;
        assert(static_cast<int>(V.extent(1)) == T && nm);
        assert(static_cast<int>(V.extent(2)) == N && nm);
        assert(static_cast<int>(V.extent(3)) == 3 && nm);
    };
    chk4(fo.pin_pos_d, "pin_pos_d");
    chk4(bo.grad_pin_pos_d, "grad_pin_pos_d");

    // ---- backward output grads ----
    // chk2(bo.grad_damping_d,        N,     "grad_damping_d");   // (1,1) per ctor
    // chk2(bo.grad_stiffness_bend_d, N - 2, "grad_kbend_d");
    // chk2(bo.grad_mass_d,           N,     "grad_mass_d");      // NOTE: per-VERTEX (N), not 3N

    // ---- forward/backward consistency: backward must run on the SAME forward ----
    assert(fo.x_hist_d.data() != nullptr && "backward() before forward()");
    assert(fo.pin_pos_d.data() != nullptr && "pin_pos not stashed -> forward() not called");
    assert(fo.x0_d.data() != nullptr && "x0 not stashed -> forward() not called");

    // ---- helper-built Views: leading dim + non-null only.
    //      TODO pin exact inner extents against the helper defs (not visible here):
    //        A_inv_d              : build_A_inv          -> likely (1, ?, ?) system inverse
    //        penaltyDt_stretch_d  : build_penaltyDt_stretch_batch(1,N,N-1,..)
    //        D_bend_d             : build_D_bend_batch(1,N)
    //        W_bend_d             : build_W_bend_batch(1,N,bend_pen)
    assert(A_inv_d.extent(0) == N && A_inv_d.extent(1) == N && "A_inv_d not N x N");
    assert(penaltyDt_stretch_d.data() != nullptr && penaltyDt_stretch_d.extent(0) == 1 &&
           "penaltyDt_stretch_d");
    assert(D_bend_d.data() != nullptr && D_bend_d.extent(0) == 1 && "D_bend_d");
    assert(W_bend_d.data() != nullptr && W_bend_d.extent(0) == 1 && "W_bend_d");
    // b : allocate_backward_scratch(1,N,gmres_m,...) — struct, assert its member views if exposed.

    Kokkos::View<double ***> loss_grad_per_step_d("loss_per_step", 1, r.T, N * 3);
    auto loss_grad_per_step_h = Kokkos::create_mirror_view(loss_grad_per_step_d);
    for (int i = 0; i < r.T; ++i)
        for (int j = 0; j < N * 3; ++j)
            loss_grad_per_step_h(0, i, j) = loss_per_grad_step(i, j);
    Kokkos::deep_copy(loss_grad_per_step_d, loss_grad_per_step_h);

    admm_backward(fo.x_hist_d, fo.y_hist_d, fo.z_hist_d, fo.dual_hist_d, fo.z_bend_hist_d,
                  fo.dual_bend_hist_d, r.mass_d, r.L0_d, r.kstretch_d, r.stretch_pen_d, r.dt,
                  r.damping_d, fo.pin_pos_d, r.is_pinned_d, A_inv_d, r.admm_iter, r.T,
                  loss_grad_per_step_d, penaltyDt_stretch_d,
                  Kokkos::View<double ***>() /*seems like a dead param*/, D_bend_d, W_bend_d,
                  bo.grad_pin_pos_d, bo.grad_damping_d, bo.grad_stiffness_bend_d, bo.grad_mass_d,
                  r.gmres_m, r.gmres_restart, r.gmres_tol, r.kbend_d, r.bend_pen, false, true, true,
                  fo.x0_d, fo.v0_d, b);
}

Eigen::MatrixXd ADMMRopeModel::rope_hist() const {
    const int N = r.rest_pos_d.extent(1) / 3;
    Eigen::MatrixXd hist(r.T, N * 3);
    auto x_hist_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.x_hist_d);
    auto y_hist_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.y_hist_d);
    auto z_hist_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fo.z_hist_d);
    for (int i = 0; i < r.T; ++i) {
        for (int j = 0; j < N * 3; ++j) {
            hist(i, j) = x_hist_h(0, i, j);
        }
    }
    return hist;
}

Eigen::MatrixXd ADMMRopeModel::grad() const {
    const int npin = r.pinned_idx.size();
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bo.grad_pin_pos_d);
    Eigen::MatrixXd grad(r.T, npin * 3);
    for (int i = 0; i < r.T; ++i)
        for (int j = 0; j < npin; ++j)
            for (int k = 0; k < 3; ++k)
                grad(i, j * 3 + k) = h(0, i, r.pinned_idx[j], k); // vtx = pinned_idx[j]
    return grad;
}