#pragma once

#include <Eigen/Dense>
#include <Kokkos_Core_fwd.hpp>
#include <tuple>
#include <vector>
#include <Kokkos_Core.hpp>
// #include "admm_forward.h"
#include "admm_backward.h"


struct RopeParams {
    Eigen::MatrixXd rest_pos; // (N, 3) baseline for stretching, bending calculations
    Eigen::VectorXd mass, damping, kstretch, kbend;
    std::vector<int> pinned_idx;
};

// Wrapper class for continuous-time rope residual dynamics, mostly for convenience and assembling
// Elastic jacobian/hessian
struct DynamicRopeModel {

    RopeParams r;
    Eigen::VectorXd rest_length, rest_curv;

    DynamicRopeModel(const RopeParams r);

    // Calculates and returns gradient in format (free, pinned)
    // Stretching, Bending: (3N)
    // r is current rope state, mode: 0 -> stretching, 1 -> bending
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> elastic_grad(const Eigen::Ref<const Eigen::MatrixXd> & r, int mode) const;

    // Calculates and returns hessian in format (Kff, Kfp, Kpp)
    // Stretching, Bending: (3N, 3N)
    // r is current rope state, mode: 0 -> stretching, 1 -> bending
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> elastic_hess(const Eigen::Ref<const Eigen::MatrixXd> & r, int mode) const;

    // Helper for (free, pinned) splitting
    std::tuple<std::vector<int>, std::vector<int>> split() const;

};

// All Kokkos is wrapped inside ADMM, everything outside is Eigen
struct ADMMParams {
    Eigen::VectorXd rest_pos; // (3N)
    Eigen::VectorXd mass, damping, kstretch, kbend;
    std::vector<int> pinned_idx;

    double dt;
    int T;

    // ADMM params
    int DEFAULT_BATCH_SIZE = 1, gmres_m = 150, gmres_restart = 60, admm_iter = 500;
    double gmres_tol = 1e-8, stretch_pen = 1e3, bend_pen = 0.0;
};

// Discrete-time rope dynamics for diffADMM integration
struct ADMMRopeModel {

    // Includes more internal things, diffADMM is batched but it is not used
    struct _ADMMParams {
        Kokkos::View<double**> rest_pos_d; // (1, 3N) baseline for stretching, bending calculations
        Kokkos::View<double**> mass_d, damping_d, kstretch_d, kbend_d; // (1, N), (1, N), (1, N-1), (1, N-2)
        std::vector<int> pinned_idx; // (npin)
        Kokkos::View<bool*> is_pinned_d; // (N)
        Kokkos::View<double**> L0_d, rest_curv_d; // (1, N-1), (1, N-2)
        Kokkos::View<double**> stretch_pen_d;

        double dt;
        int T;

        // ADMM params
        int DEFAULT_BATCH_SIZE = 1, gmres_m = 150, gmres_restart = 60, admm_iter = 500;
        double gmres_tol = 1e-8, stretch_pen = 1e3, bend_pen = 0.0;
    };
    struct FwdOut {
        Kokkos::View<double ***> x_hist_d, y_hist_d, z_hist_d, dual_hist_d, z_bend_hist_d, dual_bend_hist_d;
        Kokkos::View<double **> A_inv_d;
        // Necessary for bwd to work
        Kokkos::View<double **> /*rest_curv_d,*/ x0_d, v0_d;
        Kokkos::View<double****> pin_pos_d;
    };
    struct BwdOut {
        // Lagrangian
        Kokkos::View<double****> grad_pin_pos_d;      // (B, T, n_pins, 3)

        // Sysid, not sure why this would be necessary
        Kokkos::View<double**>   grad_damping_d;      // (1,) or (B,1)  dL/d(damping)  [batch-avg, 1/B]
        Kokkos::View<double**>   grad_stiffness_bend_d;// (B, n_bends)  dL/d(k_bend)
        Kokkos::View<double**>   grad_mass_d;         // (B, N)         dL/d(mass)
    };
    
    _ADMMParams r;
    Kokkos::View<double**> A_inv_d;
    Kokkos::View<double ***> penaltyDt_stretch_d, D_bend_d, W_bend_d;
    Kokkos::View<double*> rest_length, rest_curv;
    BackwardScratch b;

    // Vals are cached and drawn from when necessary (prevents loss but I also dont want to
    // convert everything)
    FwdOut fo;
    BwdOut bo;

    ADMMRopeModel(ADMMParams r);

    void forward(Eigen::VectorXd x0, Eigen::VectorXd v0, Eigen::MatrixXd pin_pos);
    void backward(Eigen::MatrixXd loss_per_grad_step); // T, 3N

    // Only call after necessary functions have been run, otherwise junk will be returned
    // or it will die
    Eigen::MatrixXd rope_hist() const; // (T, 3N)
    Eigen::MatrixXd grad() const;
};
