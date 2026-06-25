#include "diffadmm_wrapper.h"

#include "admm_backward.h"
#include "admm_forward.h"

/*
Structure:

namespace diffadmm_wrapper {

    // Anonymous namespace containing non-public facing methods
    namespace {

        Internal interace structs...

        fwd_in build_fwd_in(...) // Input proc for fwd
        fwd_out _forward(...) // Runs internal diffadmm_forward
        Kokkos::View<double ***> _compute_jxu(...) // Computes Jxu
    }

    Public facing methods (see diffadmm_wrapper.h)

}
*/

namespace diffadmm_wrapper {
    // Internals
    // IO of diffadmm interfacing organized into structs
    namespace {

        // ---------- Internal IO structs ----------

        struct FwdIn {
            Kokkos::View<double **> x0, v0, mass, L0;
            Kokkos::View<double **> stiffness_stretch, penalty_stretch, stiffness_bend;
            Kokkos::View<double **> damping;
            Kokkos::View<bool *> is_pinned;
            Kokkos::View<double ****> pin_pos;
            Kokkos::View<double **> rest_curv;
        };

        struct FwdOut {
            Kokkos::View<double ***> x_hist, y_hist, z_hist, dual_hist, z_bend_hist, dual_bend_hist;
            Kokkos::View<double **> A_inv;
            Kokkos::View<double ***> penaltyDt_stretch, D_bend, W_bend;
            Kokkos::View<double **> rest_curv;
        };

        struct JxuIn {
            Kokkos::View<double ***> x_hist, y_hist, z_hist, dual_hist, z_bend_hist, dual_bend_hist;
            Kokkos::View<double **> A_inv;
            Kokkos::View<double ***> penaltyDt_stretch, D_bend, W_bend;
            Kokkos::View<double **> rest_curv, mass, L0, stiffness_stretch, penalty_stretch,
                stiffness_bend;
            Kokkos::View<bool *> is_pinned;
        };

        // ---------- Input Processing ----------

        FwdIn build_fwd_in(const NpArray &x0, const NpArray &v0, const NpArray &mass,
                           const NpArray &L0, const NpArray &stiffness_stretch,
                           const NpArray &penalty_stretch, const NpArray &stiffness_bend,
                           const std::optional<NpArray> &damping,
                           std::vector<int> pin_indices, // by value: normalized in place
                           const std::optional<NpArray> &pin_positions, int T) {
            const int B = mass.shape(0);
            const int N = mass.shape(1);
            const int Bn = (N >= 3) ? (N - 2) : 0;

            // normalize negative indices
            for (int &p : pin_indices) {
                if (p < 0) {
                    p += N;
                }
            }

            FwdIn in;

            // --- direct array conversions ---
            in.x0 = np_to_kokkos_2d(x0, "x0");
            in.v0 = np_to_kokkos_2d(v0, "v0");
            in.mass = np_to_kokkos_2d(mass, "mass");
            in.L0 = np_to_kokkos_2d(L0, "L0");
            in.stiffness_stretch = np_to_kokkos_2d(stiffness_stretch, "stiffness_stretch");
            in.penalty_stretch = np_to_kokkos_2d(penalty_stretch, "penalty_stretch");
            in.stiffness_bend = np_to_kokkos_2d(stiffness_bend, "stiffness_bend");

            in.damping = Kokkos::View<double **>("damping", B, N);
            if (damping.has_value()) {
                in.damping = np_to_kokkos_2d(damping.value(), "damping");
            } else {
                Kokkos::deep_copy(in.damping, 0.0);
            }

            // --- is_pinned mask ---
            in.is_pinned = Kokkos::View<bool *>("is_pinned", N);
            {
                auto host = Kokkos::create_mirror_view(in.is_pinned);
                for (int i = 0; i < N; ++i) {
                    host(i) = false;
                }
                for (int p : pin_indices) {
                    host(p) = true;
                }
                Kokkos::deep_copy(in.is_pinned, host);
            }

            // --- pin_pos: prescribed pinned-vertex trajectories, (B, T, N, 3).
            //     Supplied as (B, T, n_pins, 3) in pin-index order and scattered;
            //     if omitted, each pinned vertex is held at its x0 position. ---
            in.pin_pos = Kokkos::View<double ****>("pin_pos", B, T, N, 3);
            {
                auto host = Kokkos::create_mirror_view(in.pin_pos);
                Kokkos::deep_copy(host, 0.0);

                if (pin_positions.has_value()) {
                    const NpArray &P = pin_positions.value();
                    if (P.ndim() != 4 || P.shape(0) != B || P.shape(1) != T ||
                        static_cast<size_t>(P.shape(2)) != pin_indices.size() || P.shape(3) != 3) {
                        throw std::runtime_error("pin_positions must have shape (B, T, n_pins, 3)");
                    }

                    auto pp = P.unchecked<4>();
                    for (int b = 0; b < B; ++b)
                        for (size_t k = 0; k < pin_indices.size(); ++k) {
                            const int p = pin_indices[k];
                            for (int t = 0; t < T; ++t) {
                                host(b, t, p, 0) = pp(b, t, (py::ssize_t)k, 0);
                                host(b, t, p, 1) = pp(b, t, (py::ssize_t)k, 1);
                                host(b, t, p, 2) = pp(b, t, (py::ssize_t)k, 2);
                            }
                        }
                } else {
                    // hold each pinned vertex at its x0 position for all t
                    auto x0r = x0.unchecked<2>();
                    for (int b = 0; b < B; ++b)
                        for (size_t k = 0; k < pin_indices.size(); ++k) {
                            const int p = pin_indices[k];
                            for (int t = 0; t < T; ++t) {
                                host(b, t, p, 0) = x0r(b, 3 * p + 0);
                                host(b, t, p, 1) = x0r(b, 3 * p + 1);
                                host(b, t, p, 2) = x0r(b, 3 * p + 2);
                            }
                        }
                }
                Kokkos::deep_copy(in.pin_pos, host);
            }

            // --- rest_curv: |x_i - 2 x_{i+1} + x_{i+2}| per interior node ---
            in.rest_curv = Kokkos::View<double **>("rest_curv", B, Bn);
            {
                auto host = Kokkos::create_mirror_view(in.rest_curv);
                auto x0r = x0.unchecked<2>();
                for (int b = 0; b < B; ++b)
                    for (int i = 0; i < Bn; ++i) {
                        const double c0 = x0r(b, 3 * i + 0) - 2 * x0r(b, 3 * (i + 1) + 0) +
                                          x0r(b, 3 * (i + 2) + 0);
                        const double c1 = x0r(b, 3 * i + 1) - 2 * x0r(b, 3 * (i + 1) + 1) +
                                          x0r(b, 3 * (i + 2) + 1);
                        const double c2 = x0r(b, 3 * i + 2) - 2 * x0r(b, 3 * (i + 1) + 2) +
                                          x0r(b, 3 * (i + 2) + 2);
                        host(b, i) = std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
                    }
                Kokkos::deep_copy(in.rest_curv, host);
            }

            return in;
        }

        // ---------- Internal DifADMM Calls ----------

        FwdOut _forward(const FwdIn &in, double dt, double penalty_bend, int ADMM_ITERS, int T,
                        bool bending_admm, bool stretching_admm, bool bending_as_force,
                        double admm_tol, int admm_check_interval) {
            const int B = in.mass.extent_int(0);
            const int N = in.mass.extent_int(1);
            const int E = N - 1;
            const int Bn = (N >= 3) ? (N - 2) : 0;

            auto A_inv = build_A_inv(in.mass, 1.0 / (dt * dt), in.penalty_stretch, in.is_pinned,
                                     penalty_bend, bending_admm, stretching_admm, bending_as_force,
                                     in.stiffness_bend);

            auto penaltyDt_stretch = build_penaltyDt_stretch_batch(B, N, E, in.penalty_stretch);
            auto D_bend = build_D_bend_batch(B, N);
            auto W_bend = build_W_bend_batch(B, N, penalty_bend);

            Kokkos::View<double ***> x_hist("x_hist", B, T, 3 * N);
            Kokkos::View<double ***> y_hist("y_hist", B, T, 3 * N);
            Kokkos::View<double ***> z_hist("z_hist", B, T, 3 * E);
            Kokkos::View<double ***> dual_hist("dual_hist", B, T, 3 * E);
            Kokkos::View<double ***> z_bend_hist("z_bend_hist", B, T, 3 * Bn);
            Kokkos::View<double ***> dual_bend_hist("dual_bend_hist", B, T, 3 * Bn);

            admm_forward(in.x0, in.v0, in.mass, in.L0, in.stiffness_stretch, in.penalty_stretch,
                         in.pin_pos, in.is_pinned, A_inv, dt, 1.0 / (dt * dt), in.damping,
                         ADMM_ITERS, T, x_hist, y_hist, z_hist, dual_hist, z_bend_hist,
                         dual_bend_hist, in.stiffness_bend, penalty_bend, bending_admm,
                         stretching_admm, bending_as_force, in.rest_curv, admm_tol,
                         admm_check_interval);
            Kokkos::fence("forward_fence");

            return FwdOut{x_hist,      y_hist,         z_hist,      dual_hist,
                          z_bend_hist, dual_bend_hist, A_inv,       penaltyDt_stretch,
                          D_bend,      W_bend,         in.rest_curv};
        }

        Kokkos::View<double ***> _compute_jxu(const JxuIn &in, std::vector<int> t, double dt,
                                              double penalty_bend, bool bending_admm,
                                              bool stretching_admm, bool bending_as_force,
                                              int gmres_m, int gmres_restart, double gmres_tol) {
            const int B = in.mass.extent_int(0);
            const int N = in.mass.extent_int(1);
            const int N3 = 3 * N;
            const int E = N - 1;
            const int E3 = stretching_admm ? 3 * E : 0;

            auto scratch = allocate_JUScratch(B, N, gmres_m, stretching_admm, bending_admm,
                                              bending_as_force, in.is_pinned, t.size());

            // T conversion to Kokkos
            Kokkos::View<int *> t_d("t_d", t.size());
            auto t_h = Kokkos::create_mirror_view(t_d);
            for (int i = 0; i < t.size(); ++i) {
                t_h(i) = t[i];
            }
            Kokkos::deep_copy(t_d, t_h);

            if ((bending_admm || bending_as_force) && scratch.rest_curv_d.extent(1) > 0) {
                Kokkos::deep_copy(scratch.rest_curv_d, in.rest_curv);
            }

            Kokkos::View<double ***> J_xu("J_xu", (int)t.size() * B, N3, scratch.n_u);

            admm_control_dense_state(
                in.x_hist, in.y_hist, in.z_hist, in.dual_hist, in.z_bend_hist, in.dual_bend_hist,
                t_d, 1.0 / (dt * dt), in.is_pinned, in.mass, in.A_inv, in.L0, in.stiffness_stretch,
                in.penalty_stretch, in.stiffness_bend, penalty_bend, bending_admm, stretching_admm,
                bending_as_force, in.penaltyDt_stretch, in.D_bend, in.W_bend, B, N, N3, E, E3,
                scratch, gmres_m, gmres_restart, gmres_tol, J_xu);

            Kokkos::fence("jxu_fence");
            return J_xu;
        }
    } // namespace

    // ---------- Pybind Exposed Methods ----------

    py::dict forward(NpArray x0, NpArray v0, NpArray mass, NpArray L0, NpArray stiffness_stretch,
                     NpArray penalty_stretch, NpArray stiffness_bend,
                     std::optional<NpArray> damping, std::vector<int> pin_indices,
                     std::optional<NpArray> pin_positions, double dt, double penalty_bend,
                     int ADMM_ITERS, int T, bool bending_admm, bool stretching_admm,
                     bool bending_as_force, double admm_tol, int admm_check_interval) {
        FwdIn in = build_fwd_in(x0, v0, mass, L0, stiffness_stretch, penalty_stretch,
                                stiffness_bend, damping, std::move(pin_indices), pin_positions, T);

        FwdOut out = _forward(in, dt, penalty_bend, ADMM_ITERS, T, bending_admm, stretching_admm,
                              bending_as_force, admm_tol, admm_check_interval);

        py::dict d;
        d["x_hist"] = kokkos_to_np(out.x_hist);
        d["y_hist"] = kokkos_to_np(out.y_hist);
        d["z_hist"] = kokkos_to_np(out.z_hist);
        d["dual_hist"] = kokkos_to_np(out.dual_hist);
        d["z_bend_hist"] = kokkos_to_np(out.z_bend_hist);
        d["dual_bend_hist"] = kokkos_to_np(out.dual_bend_hist);
        d["A_inv"] = kokkos_to_np(out.A_inv);
        d["penaltyDt_stretch"] = kokkos_to_np(out.penaltyDt_stretch);
        d["D_bend"] = kokkos_to_np(out.D_bend);
        d["W_bend"] = kokkos_to_np(out.W_bend);
        d["rest_curv"] = kokkos_to_np(out.rest_curv);
        return d;
    }

    py::array_t<double>
    compute_jxu(NpArray x_hist, NpArray y_hist, NpArray z_hist, NpArray dual_hist,
                NpArray z_bend_hist, NpArray dual_bend_hist, NpArray mass, NpArray A_inv,
                NpArray L0, NpArray stiffness_stretch, NpArray penalty_stretch,
                NpArray stiffness_bend, NpArray penaltyDt_stretch, NpArray D_bend, NpArray W_bend,
                NpArray rest_curv, std::vector<int> pin_indices, std::vector<int> t, double dt,
                double penalty_bend, bool bending_admm, bool stretching_admm, bool bending_as_force,
                int gmres_m, int gmres_restart, double gmres_tol) {
        const int N = mass.shape(1);

        JxuIn in;

        in.x_hist = np_to_kokkos_3d(x_hist, "x_hist");
        in.y_hist = np_to_kokkos_3d(y_hist, "y_hist");
        in.z_hist = np_to_kokkos_3d(z_hist, "z_hist");
        in.dual_hist = np_to_kokkos_3d(dual_hist, "dual_hist");
        in.z_bend_hist = np_to_kokkos_3d(z_bend_hist, "z_bend_hist");
        in.dual_bend_hist = np_to_kokkos_3d(dual_bend_hist, "dual_bend_hist");
        in.A_inv = np_to_kokkos_2d(A_inv, "A_inv");
        in.penaltyDt_stretch = np_to_kokkos_3d(penaltyDt_stretch, "penaltyDt_stretch");
        in.D_bend = np_to_kokkos_3d(D_bend, "D_bend");
        in.W_bend = np_to_kokkos_3d(W_bend, "W_bend");
        in.rest_curv = np_to_kokkos_2d(rest_curv, "rest_curv");
        in.mass = np_to_kokkos_2d(mass, "mass");
        in.L0 = np_to_kokkos_2d(L0, "L0");
        in.stiffness_stretch = np_to_kokkos_2d(stiffness_stretch, "stiffness_stretch");
        in.penalty_stretch = np_to_kokkos_2d(penalty_stretch, "penalty_stretch");
        in.stiffness_bend = np_to_kokkos_2d(stiffness_bend, "stiffness_bend");

        in.is_pinned = Kokkos::View<bool *>("is_pinned", N);
        {
            auto host = Kokkos::create_mirror_view(in.is_pinned);
            for (int i = 0; i < N; ++i) {
                host(i) = false;
            }
            for (int p : pin_indices) {
                host(p < 0 ? p + N : p) = true;
            }
            Kokkos::deep_copy(in.is_pinned, host);
        }

        auto J_xu = _compute_jxu(in, t, dt, penalty_bend, bending_admm, stretching_admm,
                                 bending_as_force, gmres_m, gmres_restart, gmres_tol);
        return kokkos_to_np(J_xu);
    }

    py::array_t<double>
    forwards_with_jxu(NpArray x0, NpArray v0, NpArray mass, NpArray L0, NpArray stiffness_stretch,
                      NpArray penalty_stretch, NpArray stiffness_bend,
                      std::optional<NpArray> damping, std::vector<int> pin_indices,
                      std::optional<NpArray> pin_positions, double dt, double penalty_bend,
                      int ADMM_ITERS, int T, bool bending_admm, bool stretching_admm,
                      bool bending_as_force, double admm_tol, int admm_check_interval,
                      std::vector<int> t, int gmres_m, int gmres_restart, double gmres_tol) {
        FwdIn fin = build_fwd_in(x0, v0, mass, L0, stiffness_stretch, penalty_stretch,
                                 stiffness_bend, damping, std::move(pin_indices), pin_positions, T);

        FwdOut fout = _forward(fin, dt, penalty_bend, ADMM_ITERS, T, bending_admm, stretching_admm,
                               bending_as_force, admm_tol, admm_check_interval);

        // Hand the already-resident device views straight through; no
        // re-conversion.
        JxuIn jin;
        jin.x_hist = fout.x_hist;
        jin.y_hist = fout.y_hist;
        jin.z_hist = fout.z_hist;
        jin.dual_hist = fout.dual_hist;
        jin.z_bend_hist = fout.z_bend_hist;
        jin.dual_bend_hist = fout.dual_bend_hist;
        jin.A_inv = fout.A_inv;
        jin.penaltyDt_stretch = fout.penaltyDt_stretch;
        jin.D_bend = fout.D_bend;
        jin.W_bend = fout.W_bend;
        jin.rest_curv = fout.rest_curv;
        jin.mass = fin.mass;
        jin.L0 = fin.L0;
        jin.stiffness_stretch = fin.stiffness_stretch;
        jin.penalty_stretch = fin.penalty_stretch;
        jin.stiffness_bend = fin.stiffness_bend;
        jin.is_pinned = fin.is_pinned;

        auto J_xu = _compute_jxu(jin, t, dt, penalty_bend, bending_admm, stretching_admm,
                                 bending_as_force, gmres_m, gmres_restart, gmres_tol);
        return kokkos_to_np(J_xu);
    }
} // namespace diffadmm_wrapper