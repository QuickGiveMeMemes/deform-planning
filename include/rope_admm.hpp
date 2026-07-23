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
        void fillCache(const Eigen::Ref<const Eigen::MatrixXd> &Q, double T) override {
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
            // Smooth interp 0 -> 1
            for (int t = 0; t < rope_.r.T; ++t)
                loss_grad_per_step.row(t) = ((t == rope_.r.T - 1) ? k : std::max(k / 10.0, 1.0)) * (pin_.xhist_.row(t) - xf_.transpose());
            const auto t2 = std::chrono::steady_clock::now();
            rope_.backward(loss_grad_per_step); // TODO configure from ctor for obstacles
            const auto t3 = std::chrono::steady_clock::now();
            const double timing = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }

        double value() const override {
            double s = 0.0;
            for (int t = 0; t < rope_.r.T; ++t) {
                const double w = (t == rope_.r.T-1) ? k : std::max(k/10.0, 1.0);
                s += w/2.0 * (pin_.xhist_.row(t).transpose() - xf_).squaredNorm();
            }
            return s;
        }

        void gradient(Eigen::MatrixXd &gTau) const override {
            const int npin = pin_.vertex.size();
            auto rope_grad = rope_.grad();
            gTau = Eigen::MatrixXd::Zero(nq_ + nv_ + nv_, rope_.r.T);
            for (int t = 0; t < rope_.r.T; ++t)
                gTau.col(t).head(nv_).noalias() =
                    pin_.Jp[t].transpose() *
                    static_cast<Eigen::VectorXd>(rope_grad.row(t)) ; // J_p^T dL/dp
            // gTau *= k;
            // std::fprintf(stderr, "  [rope] max|gTau|=%.3e finite=%d\n",
            //  gTau.cwiseAbs().maxCoeff(), (int)gTau.allFinite());
        }
        // No hessian

        // GN surrogate: k * Jp^T Jp per tau (rope sensitivity d x_f/d p_t ~ I). SPD, one k.
        // Rope itself not factored in
        void hessian(std::vector<Eigen::MatrixXd>& A) const override {
            A.assign(rope_.r.T, Eigen::MatrixXd::Zero(nv_, nv_));
            for (int t = 0; t < rope_.r.T; ++t)
                A[t].noalias() = k * pin_.Jp[t].transpose() * pin_.Jp[t];   // nv_ x nv_
        }
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
            // std::cout << "Control cost: " << value(P, d, c) << '\n';
            g_P.setZero();
            g_P.segment(d.offQ(), nq_arm_) = c.dtau_dq.transpose() * u;
            g_P.segment(d.offV(), nv_arm_) = c.dtau_dv.transpose() * u;
            g_P.segment(d.offA(), nv_arm_) = c.M * u;
        }

        void addHessian(const NodeQuantities &P, const NodeDims &d, const ModelEvalCache &c,
                          Eigen::MatrixXd& H_P) const override {
            // GN Hessian of (1/2)||u||^2 with u = tau(q,v,a):  H = J_tau^T J_tau
            const auto &Jq = c.dtau_dq;   // nv x nq_arm
            const auto &Jv = c.dtau_dv;   // nv x nv_arm
            const auto &Ja = c.M;         // nv x nv_arm  (du/da = M)

            const int oq = d.offQ(), ov = d.offV(), oa = d.offA();

            H_P.block(oq, oq, nq_arm_, nq_arm_).noalias() += Jq.transpose() * Jq;
            H_P.block(ov, ov, nv_arm_, nv_arm_).noalias() += Jv.transpose() * Jv;
            H_P.block(oa, oa, nv_arm_, nv_arm_).noalias() += Ja.transpose() * Ja;

            const Eigen::MatrixXd Hqv = Jq.transpose() * Jv;
            const Eigen::MatrixXd Hqa = Jq.transpose() * Ja;
            const Eigen::MatrixXd Hva = Jv.transpose() * Ja;

            H_P.block(oq, ov, nq_arm_, nv_arm_).noalias() += Hqv;
            H_P.block(ov, oq, nv_arm_, nq_arm_).noalias() += Hqv.transpose();

            H_P.block(oq, oa, nq_arm_, nv_arm_).noalias() += Hqa;
            H_P.block(oa, oq, nv_arm_, nq_arm_).noalias() += Hqa.transpose();

            H_P.block(ov, oa, nv_arm_, nv_arm_).noalias() += Hva;
            H_P.block(oa, ov, nv_arm_, nv_arm_).noalias() += Hva.transpose();
        }

        int nq_arm_, nv_arm_;
        std::string name_;
    };
    struct FramePinConstraint : public ConstraintBase {
        FramePinConstraint(const RobotModel& m, std::vector<PinSpec> pins,
                            Eigen::Matrix3Xd tgt, std::string name = "frame_pin")
            : nv_arm_(m.nv()), tgt_(std::move(tgt)), name_(std::move(name)) {
            std::sort(pins.begin(), pins.end(),
                    [](const PinSpec& a, const PinSpec& b) { return a.vertex < b.vertex; });
            std::unordered_map<std::string, int> slotOf;
            for (int i = 0; i < m.nMonitors(); ++i) slotOf[m.monitorName(i)] = i;
            for (const PinSpec& p : pins) {
                auto it = slotOf.find(p.frameName);
                if (it == slotOf.end())
                    throw std::runtime_error("FramePinConstraint: '" + p.frameName +
                                            "' is not a registered monitor frame");
                pinSlot_.push_back(it->second);
            }
            if (static_cast<int>(pinSlot_.size()) != tgt_.cols())
                throw std::invalid_argument("FramePinConstraint: pins/targets size mismatch");
        }

        const std::string& name() const override { return name_; }
        int rows(const NodeDims&) const override { return 3 * static_cast<int>(pinSlot_.size()); }
        Sense sense() const override { return Sense::Equality; }
        unsigned modelNeeds() const override { return kMonitorPos; }

        void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                    Eigen::Ref<Eigen::VectorXd> g) const override {
            for (int j = 0; j < static_cast<int>(pinSlot_.size()); ++j)
                g.segment<3>(3 * j) = c.monPos[pinSlot_[j]] - tgt_.col(j);
        }

        void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                        MatRef J) const override {
            J.setZero();
            // arm q block only: with extraDims, d.nq = nq_arm + 3*nfree, and FK depends
            // on the arm DOFs alone (rope vertices are further along the same q field).
            for (int j = 0; j < static_cast<int>(pinSlot_.size()); ++j)
                J.block(3 * j, d.offQ(), 3, nv_arm_) = c.monJ[pinSlot_[j]];
        }

        std::vector<int>  pinSlot_;
        Eigen::Matrix3Xd  tgt_;
        int               nv_arm_;
        std::string       name_;
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

            std::vector<std::string> pinned;
            pinned.reserve(rin.pinned_idx.size());

            for (auto &p : pins) {
                pinned.push_back(p.frameName);
            }

            model_ =
                std::make_shared<RobotModel>(RobotModel::fromUrdf(armConfig(urdfPath, pinned, gravity)));
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
            
            // Hardcoded to rope + endpoints currently, unlike other parts
            Eigen::Matrix3Xd goal(3, 2);
            goal.col(0) = xf_r.row(0).transpose();                // rope vertex 0   -> arm1 EE
            goal.col(1) = xf_r.row(xf_r.rows() - 1).transpose();  // rope vertex n-1 -> arm2 EE
            std::cout << xf_r << std::endl;
            std::cout << goal << std::endl;

            ms.attachments.push_back(
                {std::make_shared<FramePinConstraint>(m, pins, goal, "frame_pin_f"),
                W::LastNode});

            // Boundary conditions
            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
                                                                      q0_arm, "pin_q0_arm", Enforce::Dirichlet),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      v0_arm, "pin_v0_arm", Enforce::Dirichlet),
                                      W::FirstNode});
            

            // qf, vf only for ref traj generation
            // ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
            //                                                           qf_arm, "pin_qf_arm"),
            //                           W::LastNode});
            

            // Mostly because a zero condition is necessary
            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      vf_arm, "pin_vf_arm"),
                                      W::LastNode});

            ms.costs.push_back(std::make_shared<SimpleArmControlCost>(m));

            spec_.modes = {std::move(ms)}; // single mode; spec_.resets stays empty
            
            Eigen::VectorXd q0(d.nq), qf(d.nq);
            q0 = q0_arm;
            qf = qf_arm;
            buildInitialGuess(q0, qf, v0_arm, vf_arm);
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
        void buildInitialGuess(const Eigen::VectorXd &q0, const Eigen::VectorXd &qf, const Eigen::VectorXd &v0, const Eigen::VectorXd &vf) {
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
            V.col(0)         = v0;
            V.col(sl.Nn - 1) = vf;
        }

        std::shared_ptr<RobotModel> model_;
        pinocchio::Model loc_m;
        ADMMParams rp;

        std::shared_ptr<const Parameterization> param_;
        MultiModeSpec spec_;
        Eigen::VectorXd y0_;
    };

} // namespace leap::examples
