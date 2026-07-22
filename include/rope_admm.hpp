#pragma once

// #include <algorithm>
#include <eigen3/Eigen/Core>
#include <memory>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

// #include "pinocchio/multibody/data.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <leap/examples/trajectory_problem.hpp>
#include <leap/model/robot_config.hpp>
#include <leap/model/robot_model.hpp>
#include <leap/problem/constraint_base.hpp>
#include <leap/problem/constraints.hpp>
#include <leap/problem/costs.hpp>
#include <leap/problem/mode_spec.hpp>
#include <leap/problem/rollout_cost.hpp>
#include <leap/spectral/spectral_ops.hpp>
#include <leap/transcription/compiled_problem.hpp>
#include <leap/transcription/layout.hpp>
#include <leap/transcription/parameterization.hpp>

#include <leap/core/node_dims.hpp>
#include <leap/core/node_quantities.hpp>

#include "rope_model.hpp"
#include "common.hpp"

namespace leap::examples {
    struct RopeCost final : public RolloutCostBase {
        struct PinCache {
            std::vector<int> vertex;
            std::vector<pinocchio::FrameIndex> pinFrame_;
            Eigen::MatrixXd p;               // (t, npin * 3)
            std::vector<Eigen::MatrixXd> Jp; // (t, npin * 3, nv)
            Eigen::MatrixXd xhist_;
        };
        ADMMRopeModel rope_;
        const pinocchio::Model &model_;
        int nq_, nv_;
        double k; // k is lsq coeff
        std::string name_;
        pinocchio::Data data_;
        PinCache pin_;
        Eigen::VectorXd x0_, v0_, xf_;

        // Due to structural limitations FK must be performed internally
        RopeCost(pinocchio::Model &m, const ADMMParams &rp, std::vector<PinSpec> pins,
                 std::string name, Eigen::MatrixXd x0, Eigen::MatrixXd v0, Eigen::MatrixXd xf,
                 double k)
            : rope_(rp), model_(m), nq_(m.nq), nv_(m.nv), name_(name), data_(m), k(k) {
            const int npin = pins.size();
            pin_.vertex.reserve(npin);
            pin_.pinFrame_.reserve(npin);
            pin_.Jp.reserve(rope_.r.T);
            pin_.p = Eigen::MatrixXd::Zero(rope_.r.T, npin * 3);
            for (int t = 0; t < rope_.r.T; ++t)
                pin_.Jp.push_back(Eigen::MatrixXd::Zero(npin * 3, nv_));
            for (int i = 0; i < npin; ++i) {
                pin_.vertex.push_back(pins[i].vertex);
                pin_.pinFrame_.push_back(model_.getFrameId(pins[i].frameName));
            }
            x0_ = x0.reshaped<Eigen::RowMajor>();
            v0_ = v0.reshaped<Eigen::RowMajor>();
            xf_ = xf.reshaped<Eigen::RowMajor>();
        }

        const std::string &name() const override { return name_; }
        unsigned modelNeeds() const override { return 0; }

        // Dumps into a file (preferably a csv)
        void dumpRope(const std::string &path) const {
            const int T = rope_.r.T, N3 = static_cast<int>(pin_.xhist_.cols());
            std::ofstream f(path);
            f << std::setprecision(10);
            f << "# T=" << T << " N=" << N3/3 << " dt=" << rope_.r.dt << " pinned=";
            for (size_t i = 0; i < pin_.vertex.size(); ++i) f << (i ? ";" : "") << pin_.vertex[i];
            f << "\n";
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < N3; ++j) f << (j ? "," : "") << pin_.xhist_(t, j);
                f << "\n";
            }
        }

        // Assumed points are [-1, ..., 1] order, Q is (t, nq + nv + na)
        void fillCache(const Eigen::Ref<const Eigen::MatrixXd> &Q, ModelEvalCache &,
                       double T) override {
            const int npin = pin_.vertex.size();
            const int N3 = rope_.r.rest_pos_d.extent(1);
            Eigen::Matrix<double, 6, Eigen::Dynamic> J6(6, model_.nv); // scratch
            for (int t = 0; t < rope_.r.T; ++t) {
                Eigen::VectorXd Q_t = Q.col(t);
                const auto q = Q_t.head(nq_);
                const auto v = Q_t.segment(nq_, nv_);
                pinocchio::computeJointJacobians(model_, data_, q);
                pinocchio::updateFramePlacements(model_, data_);
                for (int i = 0; i < npin; ++i) {
                    pin_.p.row(t).segment(i * 3, 3) = data_.oMf[pin_.pinFrame_[i]].translation();
                    J6.setZero();
                    pinocchio::getFrameJacobian(model_, data_, pin_.pinFrame_[i],
                                                pinocchio::LOCAL_WORLD_ALIGNED, J6);
                    pin_.Jp[t].block(i * 3, 0, 3, nv_) = J6.topRows<3>();
                }
            }

            // dangerous, OMP serialization for Kokkos, could blow everything up
            // const int leap_threads = omp_get_max_threads();
            // omp_set_num_threads(1);
            const auto t0 = std::chrono::steady_clock::now();
            rope_.forward(x0_, v0_, pin_.p);
            const auto t1 = std::chrono::steady_clock::now();
            pin_.xhist_ = rope_.rope_hist();
            Eigen::MatrixXd loss_grad_per_step = Eigen::MatrixXd::Zero(rope_.r.T, N3);
            Eigen::VectorXd xf_sim = pin_.xhist_.row(rope_.r.T - 1);
            loss_grad_per_step.row(rope_.r.T - 1) = k * (xf_sim - xf_);
            const auto t2 = std::chrono::steady_clock::now();
            rope_.backward(loss_grad_per_step); // TODO configure from ctor for obstacles
            const auto t3 = std::chrono::steady_clock::now();
            const double timing = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // omp_set_num_threads(leap_threads);
            // std::fprintf(stderr, "  fwd=%.3f  bwd=%.3f\n",
            //     std::chrono::duration<double>(t1-t0).count(),
            //     std::chrono::duration<double>(t3-t2).count());
            // std::fprintf(stderr, "  xhist finite=%d max|x|=%.3e\n",
            //     (int)pin_.xhist_.allFinite(), pin_.xhist_.cwiseAbs().maxCoeff());

            // // after rope_.backward():
            // auto ic = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rope_.b.inner_citer_d);
            // auto icv= Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rope_.b.inner_converged_d);
            // auto cv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rope_.b.converged_d);
            // auto bg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rope_.b.beta_gmres_d);
            // std::fprintf(stderr, "  gmres(t=0): conv=%d inner=%d iters=%d beta=%.3e\n",
            //     cv(0), icv(0), ic(0), bg(0));
            // int first_bad = -1;
            // for (int t = 0; t < rope_.r.T && first_bad < 0; ++t)
            //     if (!pin_.xhist_.row(t).allFinite()) first_bad = t;
            // std::fprintf(stderr, "  first NaN at t=%d\n", first_bad);
            // std::fprintf(stderr, "  x0 fin=%d |x0|=%.2e  v0 fin=%d  pin fin=%d max|p|=%.2e  dt=%.3g\n",
            //             (int)x0_.allFinite(), x0_.cwiseAbs().maxCoeff(),
            //             (int)v0_.allFinite(), (int)pin_.p.allFinite(),
            //             pin_.p.cwiseAbs().maxCoeff(), rope_.r.dt);
        }

        double value() const override {
            Eigen::VectorXd xf_sim = pin_.xhist_.row(rope_.r.T - 1);
            return k / 2.0 * (xf_sim - xf_).squaredNorm();
        }

        void gradient(Eigen::MatrixXd &gTau) const override {
            const int npin = pin_.vertex.size();
            auto rope_grad = rope_.grad();
            gTau = Eigen::MatrixXd::Zero(nq_ + nv_ + nv_, rope_.r.T);
            for (int t = 0; t < rope_.r.T; ++t)
                gTau.col(t).head(nv_).noalias() =
                    pin_.Jp[t].transpose() *
                    static_cast<Eigen::VectorXd>(rope_grad.row(t)); // J_p^T dL/dp
        }
        // No hessian
    };

    // Different from the one in rope, no reaction force so pins dont need to be kept track of here
    struct SimpleArmControlCost final : public CostBase {
        SimpleArmControlCost(const RobotModel &m) : nq_arm_(m.nq()), nv_arm_(m.nv()), name_("arm") {}

        unsigned modelNeeds() const override { return kRnea | kRneaDerivs; }

        double value(const NodeQuantities &P, const NodeDims &,
                     const ModelEvalCache &c) const override {
            return c.tau.squaredNorm() / 2.0;
        }

        void gradient(const NodeQuantities &P, const NodeDims &d, const ModelEvalCache &c,
                      Eigen::Ref<Eigen::VectorXd> g_P) const override {
            Eigen::VectorXd u = c.tau;
            g_P.setZero();
            g_P.segment(d.offQ(), nq_arm_) = c.dtau_dq.transpose() * u;
            g_P.segment(d.offV(), nv_arm_) = c.dtau_dv.transpose() * u;
            g_P.segment(d.offA(), nv_arm_) = c.M * u;
        }
        int nq_arm_, nv_arm_;
        std::string name_;
    };

    // Contiguous index list [first, first+count) into a per-node [q;v;a;lam;u] vector.
    // The namespace is a ugly patch, cpp currently includes both this file and the residual
    // header for FK/some things because I was lazy, TODO fix
    // namespace {
    //     inline std::vector<int> indexRange(int first, int count) {
    //         std::vector<int> v(static_cast<size_t>(count));
    //         for (int i = 0; i < count; ++i) {
    //             v[static_cast<size_t>(i)] = first + i;
    //         }
    //         return v;
    //     }
    // }

    // RobotConfig for a FIXED-BASE robot: no contacts, no monitor frames. gravity
    // defaults to z-up (0,0,-9.81) -- the RobotConfig struct default is the planar
    // walker's y-down, so an arm must set it. When actuatedJoints is empty (the
    // default), every 1-DOF joint in the URDF is treated as actuated (a fully-actuated
    // arm): a throwaway probe model with no actuation is valid and lets us read
    // jointNames() (q/v order, "universe" excluded) to fill the list generically.
    // Pass an explicit list for an under-actuated fixed-base robot.
    inline RobotConfig simpleArmConfig(const std::string &urdfPath,
                                 const Eigen::Vector3d &gravity = Eigen::Vector3d(0.0, 0.0, -9.81),
                                 std::vector<std::string> actuatedJoints = {}) {
        RobotConfig cfg;
        cfg.urdfPath = urdfPath;
        cfg.gravity = gravity;
        if (actuatedJoints.empty()) {
            RobotConfig probe;
            probe.urdfPath = urdfPath;
            probe.gravity = gravity; // actuatedJoints empty => nu=0 probe, valid
            const RobotModel m0 = RobotModel::fromUrdf(probe);
            actuatedJoints = m0.jointNames();
        }
        cfg.actuatedJoints = std::move(actuatedJoints);
        return cfg;
    }

    class ADMMProblem final : public TrajectoryProblem {
      public:
        ADMMProblem(const std::string &urdfPath, const Eigen::Vector3d &gravity,
                    Eigen::VectorXd q0_arm, Eigen::VectorXd v0_arm, Eigen::MatrixXd x0_r,
                    Eigen::MatrixXd v0_r, Eigen::VectorXd qf_arm, Eigen::VectorXd vf_arm,
                    Eigen::MatrixXd xf_r, // Eigen::MatrixXd vf_r,
                    ADMMParams &rin, std::vector<PinSpec> pins, double T, int degree, double k)
            : rp(rin) {
            if (degree < 4)
                throw std::invalid_argument("Degree must be >= 4 (need interior DOF beyond the "
                                            "two-point q/v boundary conditions)");
            if (T <= 0.0)
                throw std::invalid_argument("T must be > 0");

            const int n_rope = rin.rest_pos.rows();
            const int N3 = n_rope * 3; // xyz

            model_ =
                std::make_shared<RobotModel>(RobotModel::fromUrdf(simpleArmConfig(urdfPath, gravity)));
            param_ = std::make_shared<FullStateParam>();

            const RobotModel &m = *model_;
            const int nq_arm = m.nq(), nv_arm = m.nv();
            const ContactSet none{};

            // nlam, nu should be 0
            NodeDims d;
            d.nq = m.nq();
            d.nv = m.nv();
            d.na = m.nv();

            using W = ConstraintAttachment::Where;
            ModeSpec ms;
            ms.degree = degree;
            ms.T = T;
            ms.contacts = none;

            pinocchio::urdf::buildModel(urdfPath, loc_m);

            std::vector<double> taus(rin.T);
            for (int i = 0; i < rin.T; ++i)
                taus[i] = -1.0 + i * 2.0 / (rin.T - 1.0);

            // Node-enforced rope dynamics
            ropeCost_ = std::make_shared<RopeCost>(loc_m, rp, pins, "admm rope", x0_r, v0_r, xf_r, k);
            ms.rollout.push_back({ropeCost_, taus});

            // Boundary conditions
            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
                                                                      q0_arm, "pin_q0_arm"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      v0_arm, "pin_v0_arm"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
                                                                      qf_arm, "pin_qf_arm"),
                                      W::LastNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      vf_arm, "pin_vf_arm"),
                                      W::LastNode});

            ms.costs.push_back(std::make_shared<SimpleArmControlCost>(m));

            spec_.modes = {std::move(ms)}; // single mode; spec_.resets stays empty
            
            Eigen::VectorXd q0(d.nq), qf(d.nq);
            q0 = q0_arm;
            qf = qf_arm;
            buildInitialGuess(q0, qf);
        }

        std::shared_ptr<RobotModel> model() const override { return model_; }
        std::shared_ptr<const Parameterization> parameterization() const override { return param_; }
        const MultiModeSpec &spec() const override { return spec_; }
        const Eigen::VectorXd &initialGuess() const override { return y0_; }

        std::shared_ptr<RopeCost> ropeCost_;
      private:
        // Linear pose interpolation q0 -> qf across the segment's CGL time grid, with
        // spectral V = a_scale Q Dt, A = a_scale^2 Q D2t; U and duals zero. Mirrors the
        // Go2 builder's initial guess for a single segment.
        void buildInitialGuess(const Eigen::VectorXd &q0, const Eigen::VectorXd &qf) {
            const CompiledProblem prob = CompiledProblem::compile(model_, spec_, param_);
            const Layout &L = prob.layout();
            y0_.setZero(L.n());
            const SegmentLayout &sl = L.seg[0];
            const SpectralOps &ops = prob.ops(0);
            const Eigen::VectorXd t = prob.grid(0).timeGrid(sl.T);
            Eigen::Map<Eigen::MatrixXd> Q(y0_.data() + sl.x0 + sl.off[kFieldQ], sl.dims.nq, sl.Nn);
            for (int k = 0; k < sl.Nn; ++k)
                Q.col(k) = q0 + (qf - q0) * (t(k) / sl.T);
            // Physical Q -> decision Q via the column pullback (ones under FullStateParam).
            Eigen::VectorXd sigma;
            param_->qColumnScales(sl, ops, sigma);
            for (int k = 0; k < sl.Nn; ++k)
                Q.col(k) /= sigma(k);
            Eigen::Map<Eigen::MatrixXd> V(y0_.data() + sl.x0 + sl.off[kFieldV], sl.dims.nv, sl.Nn);
            Eigen::Map<Eigen::MatrixXd> A(y0_.data() + sl.x0 + sl.off[kFieldA], sl.dims.na, sl.Nn);
            V = ops.a_scale * Q * ops.Dt;
            A = ops.a_scale * ops.a_scale * Q * ops.D2t;
        }

        std::shared_ptr<RobotModel> model_;
        pinocchio::Model loc_m;
        ADMMParams rp;

        std::shared_ptr<const Parameterization> param_;
        MultiModeSpec spec_;
        Eigen::VectorXd y0_;
    };

} // namespace leap::examples
