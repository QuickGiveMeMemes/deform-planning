#pragma once

#include <Kokkos_Core.hpp>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <type_traits>
#include <vector>

namespace diffadmm_wrapper {
    namespace py = pybind11;

    using NpArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    // ---------- Numpy <-> Kokkos utils ----------
    //
    // Everything here is TMP
    // TODO: current conversion only supports float64 numpy arrays. Generalize.

    // Alloc new Kokkos::View memory
    template <typename View, std::size_t... I>
    View alloc(const std::string &name, const py::ssize_t *s, std::index_sequence<I...>) {
        return View(name, static_cast<size_t>(s[I])...);
    }
    // Wrap existing python memory in Kokkos::View
    template <typename View, std::size_t... I>
    View alloc(void *ptr, const py::ssize_t *s, std::index_sequence<I...>) {
        auto cast_ptr = static_cast<typename View::value_type *>(ptr);
        return View(cast_ptr, static_cast<size_t>(s[I])...);
    }

    // LayoutRight unmanaged Kokkos::View for conversion to/from numpy
    template <typename View>
    using HostLR = Kokkos::View<typename View::data_type, Kokkos::LayoutRight, Kokkos::HostSpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    template <typename View> View np_to_kokkos(const NpArray &a, const std::string &name) {
        static_assert(std::is_same_v<typename View::value_type, double>,
                      "np_to_kokkos input is fixed to float64");

        constexpr std::size_t R = View::rank();
        auto buf = a.request();
        if (static_cast<std::size_t>(buf.ndim) != R) {
            throw std::runtime_error("rank mismatch");
        }

        View device = alloc<View>(name, buf.shape.data(), std::make_index_sequence<R>{});

        HostLR<View> host_np =
            alloc<HostLR<View>>(buf.ptr, buf.shape.data(), std::make_index_sequence<R>{});

        auto mirror =
            Kokkos::create_mirror_view(device); // device layout (LayoutLeft expected), host space
        Kokkos::deep_copy(mirror, host_np);     // converts into device layout
        Kokkos::deep_copy(device, mirror);
        return device;
    }

    template <typename View, std::size_t... I>
    std::vector<py::ssize_t> kokkos_shape(const View &a, std::index_sequence<I...>) {
        return {static_cast<py::ssize_t>(a.extent(I))...};
    }

    template <typename View, std::size_t... I>
    HostLR<View> kokkos_wrap(typename View::value_type *ptr, const View &v,
                             std::index_sequence<I...>) {
        return HostLR<View>(ptr, v.extent(I)...);
    }

    template <typename View> py::array_t<typename View::value_type> kokkos_to_np(const View &a) {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
        constexpr std::size_t R = View::rank();

        auto size = kokkos_shape(a, std::make_index_sequence<R>{});
        py::array_t<typename View::value_type> np_a(size);

        auto wrap = kokkos_wrap(np_a.mutable_data(), a, std::make_index_sequence<R>{});
        Kokkos::deep_copy(wrap, mirror);

        return np_a;
    }

    // Convenience wrappers

    inline Kokkos::View<double **> np_to_kokkos_2d(const NpArray &a, const std::string &name) {
        return np_to_kokkos<Kokkos::View<double **>>(a, name);
    }
    inline Kokkos::View<double ***> np_to_kokkos_3d(const NpArray &a, const std::string &name) {
        return np_to_kokkos<Kokkos::View<double ***>>(a, name);
    }

    // ---------- Exposed methods ----------

    // TODO if this pybind becomes the main interface and not only the testing interface, then
    // expose the A_inv and Jxu scratch methods for efficiency. Should be easy integration into the
    // existing deformable wrapper.

    py::dict forward(NpArray x0, NpArray v0, NpArray mass, NpArray L0, NpArray stiffness_stretch,
                     NpArray penalty_stretch, NpArray stiffness_bend,
                     std::optional<NpArray> damping, std::vector<int> pin_indices,
                     std::optional<NpArray> pin_positions, double dt, double penalty_bend,
                     int ADMM_ITERS, int T, bool bending_admm, bool stretching_admm,
                     bool bending_as_force, double admm_tol, int admm_check_interval);

    py::array_t<double>
    compute_jxu(NpArray x_hist, NpArray y_hist, NpArray z_hist, NpArray dual_hist,
                NpArray z_bend_hist, NpArray dual_bend_hist, NpArray mass, NpArray A_inv,
                NpArray L0, NpArray stiffness_stretch, NpArray penalty_stretch,
                NpArray stiffness_bend, NpArray penaltyDt_stretch, NpArray D_bend, NpArray W_bend,
                NpArray rest_curv, std::vector<int> pin_indices, std::vector<int> t, double dt,
                double penalty_bend, bool bending_admm, bool stretching_admm, bool bending_as_force,
                int gmres_m, int gmres_restart, double gmres_tol);

    py::array_t<double>
    forwards_with_jxu(NpArray x0, NpArray v0, NpArray mass, NpArray L0, NpArray stiffness_stretch,
                      NpArray penalty_stretch, NpArray stiffness_bend,
                      std::optional<NpArray> damping, std::vector<int> pin_indices,
                      std::optional<NpArray> pin_positions, double dt, double penalty_bend,
                      int ADMM_ITERS, int T, bool bending_admm, bool stretching_admm,
                      bool bending_as_force, double admm_tol, int admm_check_interval,
                      std::vector<int> t, int gmres_m, int gmres_restart, double gmres_tol);
} // namespace diffadmm_wrapper