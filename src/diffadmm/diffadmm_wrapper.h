#include <Kokkos_Core.hpp>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "admm_backward.h"
#include "admm_forward.h"

namespace py = pybind11;
using NpArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Conversion utils
inline Kokkos::View<double **> np_to_kokkos_2d(const NpArray &a, const char *name) {
  const int n0 = a.shape(0);
  const int n1 = a.shape(1);

  Kokkos::View<double **> device(std::string(name), n0, n1);

  auto host = Kokkos::create_mirror_view(device);
  auto in = a.unchecked<2>();

  for (int i = 0; i < n0; ++i) { // Maybe this can be parallelized?
    for (int j = 0; j < n1; ++j) {
      host(i, j) = in(i, j);
    }
  }
  Kokkos::deep_copy(device, host);

  return device;
}

inline Kokkos::View<double ***> np_to_kokkos_3d(const NpArray &a, const char *name) {
  const int n0 = a.shape(0), n1 = a.shape(1), n2 = a.shape(2);

  Kokkos::View<double ***> device(std::string(name), n0, n1, n2);

  auto host = Kokkos::create_mirror_view(device);
  auto in = a.unchecked<3>();

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        host(i, j, k) = in(i, j, k);
      }
    }
  }

  Kokkos::deep_copy(device, host);

  return device;
}

inline py::array_t<double> kokkos_to_numpy(const Kokkos::View<double ***> &device) {
  const int n0 = device.extent(0), n1 = device.extent(1), n2 = device.extent(2);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);

  py::array_t<double> out({(py::ssize_t)n0, (py::ssize_t)n1, (py::ssize_t)n2});

  auto o = out.mutable_unchecked<3>();

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        o(i, j, k) = host(i, j, k);
      }
    }
  }

  return out;
}

inline  py::array_t<double>
kokkos_to_numpy(const Kokkos::View<double **> &device) {
  const int n0 = device.extent(0), n1 = device.extent(1);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);

  py::array_t<double> out({(py::ssize_t)n0, (py::ssize_t)n1});

  auto o = out.mutable_unchecked<2>();

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      o(i, j) = host(i, j);
    }
  }

  return out;
}

// -------------- Internal, returns kokkos ----------------
std::tuple<Kokkos::View<double ***>, Kokkos::View<double ***>,
           Kokkos::View<double ***>, Kokkos::View<double ***>,
           Kokkos::View<double ***>, Kokkos::View<double ***>,
           Kokkos::View<double **>, Kokkos::View<double ***>,
           Kokkos::View<double ***>, Kokkos::View<double ***>,
           Kokkos::View<double **>>
inline _forward(NpArray x0, NpArray v0, NpArray mass, NpArray L0,
         NpArray stiffness_stretch, NpArray penalty_stretch,
         NpArray stiffness_bend, std::optional<NpArray> damping,
         std::vector<int> pin_indices, std::optional<NpArray> pin_velocities,
         double dt, double penalty_bend, int ADMM_ITERS, int T,
         bool bending_admm, bool stretching_admm, bool bending_as_force,
         double admm_tol, int admm_check_interval) {
  const int B = mass.shape(0);
  const int N = mass.shape(1);
  const int E = N - 1;
  const int Bn = (N >= 3) ? (N - 2) : 0;

  auto x0_d = np_to_kokkos_2d(x0, "x0");
  auto v0_d = np_to_kokkos_2d(v0, "v0");
  auto mass_d = np_to_kokkos_2d(mass, "mass");
  auto L0_d = np_to_kokkos_2d(L0, "L0");
  auto stiffness_stretch_d =
      np_to_kokkos_2d(stiffness_stretch, "stiffness_stretch");
  auto penalty_stretch_d = np_to_kokkos_2d(penalty_stretch, "penalty_stretch");
  auto stiffness_bend_d = np_to_kokkos_2d(stiffness_bend, "stiffness_bend");

  for (int &p : pin_indices) {
    if (p < 0) {
      p += N;
    }
  }

  Kokkos::View<bool *> is_pinned("is_pinned", N);
  auto pin_host = Kokkos::create_mirror_view(is_pinned);
  for (int i = 0; i < N; ++i) {
    pin_host(i) = false;
  }
  for (int p : pin_indices) {
    pin_host(p) = true;
  }

  Kokkos::deep_copy(is_pinned, pin_host);

  Kokkos::View<double ****> pin_pos("pin_pos", B, T, N, 3);
  auto pin_pos_host = Kokkos::create_mirror_view(pin_pos);
  Kokkos::deep_copy(pin_pos_host, 0.0);

  {
    auto x0r = x0.unchecked<2>();
    const double *v =
        pin_velocities.has_value() ? pin_velocities.value().data() : nullptr;
    for (int b = 0; b < B; ++b)
      for (size_t k = 0; k < pin_indices.size(); ++k) {
        const double vx = v ? v[3 * k + 0] : 0.0;
        const double vy = v ? v[3 * k + 1] : 0.0;
        const double vz = v ? v[3 * k + 2] : 0.0;

        const int p = pin_indices[k];

        for (int t = 0; t < T; ++t) {
          const double s = t * dt;
          pin_pos_host(b, t, p, 0) = x0r(b, 3 * p + 0) + s * vx;
          pin_pos_host(b, t, p, 1) = x0r(b, 3 * p + 1) + s * vy;
          pin_pos_host(b, t, p, 2) = x0r(b, 3 * p + 2) + s * vz;
        }
      }
  }
  Kokkos::deep_copy(pin_pos, pin_pos_host);

  Kokkos::View<double **> damping_d("damping", B, N);

  if (damping.has_value()) {
    damping_d = np_to_kokkos_2d(damping.value(), "damping");
  } else {
    Kokkos::deep_copy(damping_d, 0.0);
  }

  auto A_inv = build_A_inv(mass_d, 1.0 / (dt * dt), penalty_stretch_d,
                           is_pinned, penalty_bend, bending_admm,
                           stretching_admm, bending_as_force, stiffness_bend_d);

  auto penaltyDt_stretch =
      build_penaltyDt_stretch_batch(B, N, E, penalty_stretch_d);
  auto D_bend = build_D_bend_batch(B, N);
  auto W_bend = build_W_bend_batch(B, N, penalty_bend);

  Kokkos::View<double ***> x_hist("x_hist", B, T, 3 * N);
  Kokkos::View<double ***> y_hist("y_hist", B, T, 3 * N);
  Kokkos::View<double ***> z_hist("z_hist", B, T, 3 * E);
  Kokkos::View<double ***> dual_hist("dual_hist", B, T, 3 * E);
  Kokkos::View<double ***> z_bend_hist("z_bend_hist", B, T, 3 * Bn);
  Kokkos::View<double ***> dual_bend_hist("dual_bend_hist", B, T, 3 * Bn);

  Kokkos::View<double **> rest_curv("rest_curv", B, Bn);
  {
    auto rc_host = Kokkos::create_mirror_view(rest_curv);
    auto x0r = x0.unchecked<2>();
    for (int b = 0; b < B; ++b)
      for (int i = 0; i < Bn; ++i) {
        const double c0 = x0r(b, 3 * i + 0) - 2 * x0r(b, 3 * (i + 1) + 0) +
                          x0r(b, 3 * (i + 2) + 0);
        const double c1 = x0r(b, 3 * i + 1) - 2 * x0r(b, 3 * (i + 1) + 1) +
                          x0r(b, 3 * (i + 2) + 1);
        const double c2 = x0r(b, 3 * i + 2) - 2 * x0r(b, 3 * (i + 1) + 2) +
                          x0r(b, 3 * (i + 2) + 2);
        rc_host(b, i) = std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
      }
    Kokkos::deep_copy(rest_curv, rc_host);
  }

  admm_forward(x0_d, v0_d, mass_d, L0_d, stiffness_stretch_d, penalty_stretch_d,
               pin_pos, is_pinned, A_inv, dt, 1.0 / (dt * dt), damping_d,
               ADMM_ITERS, T, x_hist, y_hist, z_hist, dual_hist, z_bend_hist,
               dual_bend_hist, stiffness_bend_d, penalty_bend, bending_admm,
               stretching_admm, bending_as_force, rest_curv, admm_tol,
               admm_check_interval);
  Kokkos::fence();

  return {x_hist,      y_hist,         z_hist,   dual_hist,
          z_bend_hist, dual_bend_hist, A_inv,    penaltyDt_stretch,
          D_bend,      W_bend,         rest_curv};
}

inline py::array_t<double> _compute_jxu(
    Kokkos::View<double ***> x_hist, Kokkos::View<double ***> y_hist,
    Kokkos::View<double ***> z_hist, Kokkos::View<double ***> dual_hist,
    Kokkos::View<double ***> z_bend_hist,
    Kokkos::View<double ***> dual_bend_hist, Kokkos::View<double **> A_inv,
    Kokkos::View<double ***> penaltyDt_stretch, Kokkos::View<double ***> D_bend,
    Kokkos::View<double ***> W_bend, Kokkos::View<double **> rest_curv,
    Kokkos::View<double **> mass, Kokkos::View<double **> L0,
    Kokkos::View<double **> stiffness_stretch,
    Kokkos::View<double **> penalty_stretch,
    Kokkos::View<double **> stiffness_bend, std::vector<int> pin_indices, int t,
    double dt, double penalty_bend, bool bending_admm, bool stretching_admm,
    bool bending_as_force, int gmres_m, int gmres_restart, double gmres_tol) {
  const int B = mass.extent_int(0);
  const int N = mass.extent_int(1);
  const int N3 = 3 * N;
  const int E = N - 1;
  const int E3 = stretching_admm ? 3 * E : 0;

  Kokkos::View<bool *> is_pinned("is_pinned", N);

  auto is_pinned_host = Kokkos::create_mirror_view(is_pinned);

  for (int i = 0; i < N; ++i) {
    is_pinned_host(i) = false;
  }
  for (int p : pin_indices) {
    is_pinned_host(p < 0 ? p + N : p) = true;
  }

  Kokkos::deep_copy(is_pinned, is_pinned_host);

  auto scratch = allocate_JUScratch(B, N, gmres_m, stretching_admm,
                                    bending_admm, bending_as_force, is_pinned);

  if ((bending_admm || bending_as_force) && scratch.rest_curv_d.extent(1) > 0) {
    Kokkos::deep_copy(scratch.rest_curv_d, rest_curv);
  }

  Kokkos::View<double ***> J_xu("J_xu", B, N3, scratch.n_u);

  admm_control_dense_state(
      x_hist, y_hist, z_hist, dual_hist, z_bend_hist, dual_bend_hist, t,
      1.0 / (dt * dt), is_pinned, mass, A_inv, L0, stiffness_stretch,
      penalty_stretch, stiffness_bend, penalty_bend, bending_admm,
      stretching_admm, bending_as_force, penaltyDt_stretch, D_bend, W_bend, B,
      N, N3, E, E3, scratch, gmres_m, gmres_restart, gmres_tol, J_xu);

  Kokkos::fence();

  return kokkos_to_numpy(J_xu);
}

// ---------- Exposed methods ----------
inline  py::dict forward(NpArray x0, NpArray v0, NpArray mass, NpArray L0,
                        NpArray stiffness_stretch, NpArray penalty_stretch,
                        NpArray stiffness_bend, std::optional<NpArray> damping,
                        std::vector<int> pin_indices,
                        std::optional<NpArray> pin_velocities, double dt,
                        double penalty_bend, int ADMM_ITERS, int T,
                        bool bending_admm, bool stretching_admm,
                        bool bending_as_force, double admm_tol,
                        int admm_check_interval) {
  auto [x_hist, y_hist, z_hist, dual_hist, z_bend_hist, dual_bend_hist, A_inv,
        penaltyDt_stretch, D_bend, W_bend, rest_curv] =
      _forward(x0, v0, mass, L0, stiffness_stretch, penalty_stretch,
               stiffness_bend, damping, pin_indices, pin_velocities, dt,
               penalty_bend, ADMM_ITERS, T, bending_admm, stretching_admm,
               bending_as_force, admm_tol, admm_check_interval);
  py::dict out;

  out["x_hist"] = kokkos_to_numpy(x_hist);
  out["y_hist"] = kokkos_to_numpy(y_hist);
  out["z_hist"] = kokkos_to_numpy(z_hist);
  out["dual_hist"] = kokkos_to_numpy(dual_hist);
  out["z_bend_hist"] = kokkos_to_numpy(z_bend_hist);
  out["dual_bend_hist"] = kokkos_to_numpy(dual_bend_hist);
  out["A_inv"] = kokkos_to_numpy(A_inv);
  out["penaltyDt_stretch"] = kokkos_to_numpy(penaltyDt_stretch);
  out["D_bend"] = kokkos_to_numpy(D_bend);
  out["W_bend"] = kokkos_to_numpy(W_bend);
  out["rest_curv"] = kokkos_to_numpy(rest_curv);

  return out;
}

inline py::array_t<double> compute_jxu(
    NpArray x_hist, NpArray y_hist, NpArray z_hist, NpArray dual_hist,
    NpArray z_bend_hist, NpArray dual_bend_hist, NpArray mass, NpArray A_inv,
    NpArray L0, NpArray stiffness_stretch, NpArray penalty_stretch,
    NpArray stiffness_bend, NpArray penaltyDt_stretch, NpArray D_bend,
    NpArray W_bend, NpArray rest_curv, std::vector<int> pin_indices, int t,
    double dt, double penalty_bend, bool bending_admm, bool stretching_admm,
    bool bending_as_force, int gmres_m, int gmres_restart, double gmres_tol) {

  auto x_hist_d = np_to_kokkos_3d(x_hist, "x_hist");
  auto y_hist_d = np_to_kokkos_3d(y_hist, "y_hist");
  auto z_hist_d = np_to_kokkos_3d(z_hist, "z_hist");
  auto dual_hist_d = np_to_kokkos_3d(dual_hist, "dual_hist");
  auto z_bend_hist_d = np_to_kokkos_3d(z_bend_hist, "z_bend_hist");
  auto dual_bend_hist_d = np_to_kokkos_3d(dual_bend_hist, "dual_bend_hist");
  auto mass_d = np_to_kokkos_2d(mass, "mass");
  auto A_inv_d = np_to_kokkos_2d(A_inv, "A_inv");
  auto L0_d = np_to_kokkos_2d(L0, "L0");
  auto stiffness_stretch_d =
      np_to_kokkos_2d(stiffness_stretch, "stiffness_stretch");
  auto penalty_stretch_d = np_to_kokkos_2d(penalty_stretch, "penalty_stretch");
  auto stiffness_bend_d = np_to_kokkos_2d(stiffness_bend, "stiffness_bend");
  auto penaltyDt_stretch_d =
      np_to_kokkos_3d(penaltyDt_stretch, "penaltyDt_stretch");
  auto D_bend_d = np_to_kokkos_3d(D_bend, "D_bend");
  auto W_bend_d = np_to_kokkos_3d(W_bend, "W_bend");
  auto rest_curv_d = np_to_kokkos_2d(rest_curv, "rest_curv");

  return _compute_jxu(x_hist_d, y_hist_d, z_hist_d, dual_hist_d, z_bend_hist_d,
                      dual_bend_hist_d, A_inv_d, penaltyDt_stretch_d, D_bend_d,
                      W_bend_d, rest_curv_d, mass_d, L0_d, stiffness_stretch_d,
                      penalty_stretch_d, stiffness_bend_d, pin_indices, t, dt,
                      penalty_bend, bending_admm, stretching_admm,
                      bending_as_force, gmres_m, gmres_restart, gmres_tol);
}

inline py::array_t<double> forwards_with_jxu(
    NpArray x0, NpArray v0, NpArray mass, NpArray L0, NpArray stiffness_stretch,
    NpArray penalty_stretch, NpArray stiffness_bend,
    std::optional<NpArray> damping, std::vector<int> pin_indices,
    std::optional<NpArray> pin_velocities, double dt, double penalty_bend,
    int ADMM_ITERS, int T, bool bending_admm, bool stretching_admm,
    bool bending_as_force, double admm_tol, int admm_check_interval, int t,
    int gmres_m, int gmres_restart, double gmres_tol) {
      
  auto [x_hist, y_hist, z_hist, dual_hist, z_bend_hist, dual_bend_hist, A_inv,
        penaltyDt_stretch, D_bend, W_bend, rest_curv] =
      _forward(x0, v0, mass, L0, stiffness_stretch, penalty_stretch,
               stiffness_bend, damping, pin_indices, pin_velocities, dt,
               penalty_bend, ADMM_ITERS, T, bending_admm, stretching_admm,
               bending_as_force, admm_tol, admm_check_interval);

  auto mass_d = np_to_kokkos_2d(mass, "mass");
  auto L0_d = np_to_kokkos_2d(L0, "L0");
  auto stiffness_stretch_d =
      np_to_kokkos_2d(stiffness_stretch, "stiffness_stretch");
  auto penalty_stretch_d = np_to_kokkos_2d(penalty_stretch, "penalty_stretch");
  auto stiffness_bend_d = np_to_kokkos_2d(stiffness_bend, "stiffness_bend");

  return _compute_jxu(x_hist, y_hist, z_hist, dual_hist, z_bend_hist,
                      dual_bend_hist, A_inv, penaltyDt_stretch, D_bend, W_bend,
                      rest_curv, mass_d, L0_d, stiffness_stretch_d,
                      penalty_stretch_d, stiffness_bend_d, pin_indices, t, dt,
                      penalty_bend, bending_admm, stretching_admm,
                      bending_as_force, gmres_m, gmres_restart, gmres_tol);
}