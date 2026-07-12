#include "../../include/rope_model.hpp"
#include <Eigen/Dense>
#include <vector>

DynamicRopeModel::DynamicRopeModel(const RopeParams r) : r(std::move(r)) {

    const int N = r.rest_pos.rows();

    rest_length = Eigen::VectorXd(N - 1);
    rest_curv = Eigen::VectorXd(N - 2);

    for (int i = 0; i < N - 1; ++i) {
        // Assumes initial configuration is rest length
        rest_length(i) = (
            r.rest_pos.row(i + 1) - 
            r.rest_pos.row(i)).norm();
    }

    for (int i = 0; i < N - 2; ++i) {
        // Assumes initial configuration is rest curvature
        rest_curv(i) = (
            r.rest_pos.row(i) -
            2 * r.rest_pos.row(i + 1) +
            r.rest_pos.row(i + 2)).norm();
    }
}

// Private helper for splitting into (free, pinned)
std::tuple<std::vector<int>, std::vector<int>> split(const DynamicRopeModel & r) {
    
    const int N = r.r.rest_pos.rows();

    // Mask for efficiency
    std::vector<bool> pinned(N, false);
    for (auto i : r.r.pinned_idx)
        pinned[i] = true;

    std::vector<int> p, f;
    p.reserve(r.r.pinned_idx.size());
    f.reserve(N - r.r.pinned_idx.size());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            (pinned[i] ? p : f).push_back(3 * i + j);
        }
    }

    return {f, p};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> 
DynamicRopeModel::elastic_grad(
    const Eigen::Ref<const Eigen::MatrixXd> & R, int mode) const {
    
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
        } 
        else {
            l = 2;
            rest = rest_length(i);
            k = this->r.kstretch(i);
            g = (R.row(i + 1) - R.row(i)).transpose();
            idx = {i, i + 1, 0};
            stencil = {-1, 1, 0};
        }

        const double norm = std::max(g.norm(), 1e-9);
        const Eigen::Vector3d ghat = g / norm;
        const Eigen::Vector3d f = k * (norm - rest) * ghat; // Term k( ||g|| - rest ) ghat (before D^T matmul)

        for (int a = 0; a < l; ++a)
            grad.segment(3 * idx[a], 3) += stencil[a] * f;
    }
    
    const auto [free, pinned] = split(*this);
    return {grad(free), grad(pinned)};
}


std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
DynamicRopeModel::elastic_hess(
    const Eigen::Ref<const Eigen::MatrixXd> & R, int mode) const {
    
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
        } 
        else {
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
        const Eigen::Matrix3d A = 
            k * (gg + (1 - rest / norm) * (Eigen::Matrix3d::Identity() - gg));
        
        for (int a = 0; a < l; ++a)
            for (int b = 0; b < l; ++b) {
                K.block<3, 3>(3 * idx[a], 3 * idx[b]) += stencil[a] * stencil[b] * A;
        }
    }
    
    const auto [free, pinned] = split(*this);
    return {K(free, free), K(free, pinned)};
}

