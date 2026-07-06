#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>

#include "diffadmm_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(diffadmm, m) {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        py::module_::import("atexit").attr("register")(py::cpp_function([] {
            if (Kokkos::is_initialized())
                Kokkos::finalize();
        }));
    }

    m.def("forward", &diffadmm_wrapper::forward, py::arg("x0"), py::arg("v0"), py::arg("mass"),
          py::arg("L0"), py::arg("stiffness_stretch"), py::arg("penalty_stretch"),
          py::arg("stiffness_bend"), py::arg("damping") = py::none(),
          py::arg("pin_indices") = std::vector<int>{}, py::arg("pin_positions") = py::none(),
          py::arg("dt"), py::arg("penalty_bend"), py::arg("ADMM_ITERS"), py::arg("T"),
          py::arg("bending_admm"), py::arg("stretching_admm"), py::arg("bending_as_force"),
          py::arg("admm_tol"), py::arg("admm_check_interval"));

    m.def("compute_jac", &diffadmm_wrapper::compute_jac, py::arg("x0"), py::arg("x_hist"), py::arg("y_hist"),
          py::arg("z_hist"), py::arg("dual_hist"), py::arg("z_bend_hist"),
          py::arg("dual_bend_hist"), py::arg("mass"), py::arg("A_inv"), py::arg("damping"), py::arg("L0"),
          py::arg("stiffness_stretch"), py::arg("penalty_stretch"), py::arg("stiffness_bend"),
          py::arg("penaltyDt_stretch"), py::arg("D_bend"), py::arg("W_bend"), py::arg("rest_curv"),
          py::arg("pin_indices"), py::arg("t"), py::arg("dt"), py::arg("penalty_bend"),
          py::arg("bending_admm"), py::arg("stretching_admm"), py::arg("bending_as_force"),
          py::arg("gmres_m"), py::arg("gmres_restart"), py::arg("gmres_tol"));

    m.def("forwards_with_jac", &diffadmm_wrapper::forwards_with_jac, py::arg("x0"), py::arg("v0"),
          py::arg("mass"), py::arg("L0"), py::arg("stiffness_stretch"), py::arg("penalty_stretch"),
          py::arg("stiffness_bend"), py::arg("damping"), py::arg("pin_indices"),
          py::arg("pin_positions"), py::arg("dt"), py::arg("penalty_bend"), py::arg("ADMM_ITERS"),
          py::arg("T"), py::arg("bending_admm"), py::arg("stretching_admm"),
          py::arg("bending_as_force"), py::arg("admm_tol"), py::arg("admm_check_interval"),
          py::arg("t"), py::arg("gmres_m"), py::arg("gmres_restart"), py::arg("gmres_tol"));
}