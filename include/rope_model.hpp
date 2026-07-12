#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <Eigen/Dense>

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

    // Calculates and returns hessian in format (free, pinned)
    // Stretching, Bending: (3N, 3N)
    // r is current rope state, mode: 0 -> stretching, 1 -> bending
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> elastic_hess(const Eigen::Ref<const Eigen::MatrixXd> & r, int mode) const;
};

// Discrete-time rope dynamics for diffADMM integration
struct ADMMRopeModel {

};