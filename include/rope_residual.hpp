#pragma once

// Residual-based arm-rope optimizer. The rope dynamics residual is enforced via the
// dual ALM flow, while the arm control is minimized via least-squares
//
// Ha+C=u -> least squares 1/2 || Ha+C ||^2 in cost

#include <memory>
#include <pinocchio/multibody/fwd.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <pinocchio/fwd.hpp> 
#include <pinocchio/parsers/urdf.hpp> 
#include <pinocchio/algorithm/joint-configuration.hpp>

#include "../leap-discrete/refactored_codebase_v1/include/leap/examples/trajectory_problem.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/model/robot_config.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/model/robot_model.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/problem/constraint_base.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/problem/constraints.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/problem/costs.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/problem/mode_spec.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/spectral/spectral_ops.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/transcription/compiled_problem.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/transcription/layout.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/transcription/parameterization.hpp"

#include "../leap-discrete/refactored_codebase_v1/include/leap/core/node_dims.hpp"
#include "../leap-discrete/refactored_codebase_v1/include/leap/core/node_quantities.hpp"

#include "rope_model.hpp"

namespace leap::examples {
    
    namespace pin = pinocchio;

    struct RopeConstraint final : public ConstraintBase {
        RopeConstraint(const RobotModel &m, const RopeParams &r, std::vector<pin::FrameIndex> pinned)
            : actIdx_(m.actuatedIdx()), r(DynamicRopeModel(r)), nv_(m.nv()), name_("dyn") {}

        const std::string &name() const override { return name_; }
        int rows(const NodeDims &) const override { return nv_; }
        Sense sense() const override { return Sense::Equality; }
        unsigned modelNeeds() const override {
            return kRnea | kRneaDerivs;
        }

        void value(const NodeQuantities &P, const NodeDims &, const ModelEvalCache &c,
                   Eigen::Ref<Eigen::VectorXd> g) const override {

            

                
            g = c.gdyn; // tau - sum_c Jc' lam_c (folded by evalNode; == tau when cs empty)
            for (size_t j = 0; j < actIdx_.size(); ++j)
                g(actIdx_[j]) -= P.u(static_cast<Eigen::Index>(j));
        }

        void jacobian(const NodeQuantities &, const NodeDims &d, const ModelEvalCache &c,
                      MatRef J) const override {
            J.setZero();
            J.middleCols(d.offQ(), d.nq) = c.dgdyn_dq; // == dtau_dq when cs empty
            J.middleCols(d.offV(), d.nv) = c.dtau_dv;
            J.middleCols(d.offA(), d.na) = c.M; // dtau/da = mass matrix
            int o = 0;
            for (const ContactRef &cr : cs_.active) { // skipped when cs empty
                J.middleCols(d.offLam() + o, cr.dim) =
                    -c.footJ[static_cast<size_t>(cr.index)].transpose();
                o += cr.dim;
            }
            for (size_t j = 0; j < actIdx_.size(); ++j)
                J(actIdx_[j], d.offU() + static_cast<int>(j)) = -1.0; // -B u
        }

        // H += sum_i w_i d2 g_i / dz2 via the analytic SO-RNEA provider (empty cs =>
        // pure SO-RNEA, no contact kernel). Same shape as the frame-constraint Hessians.
        void addContractedHessian(const NodeQuantities &P, const NodeDims &d, Workspace &ws,
                                  const Eigen::Ref<const Eigen::VectorXd> &w,
                                  Eigen::MatrixXd &H) const override {
            ws.z.resize(d.nP());
            P.toVector(d, ws.z);
            ws.tapes->addDynHessian(d, cs_, ws.z, w, H);
        }

        std::vector<int> actIdx_;
        DynamicRopeModel r;
        int nv_;
        std::string name_;
    };

    // Contiguous index list [first, first+count) into a per-node [q;v;a;lam;u] vector.
    inline std::vector<int> indexRange(int first, int count) {
        std::vector<int> v(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            v[static_cast<size_t>(i)] = first + i;
        return v;
    }

    // RobotConfig for a FIXED-BASE robot: no contacts, no monitor frames. gravity
    // defaults to z-up (0,0,-9.81) -- the RobotConfig struct default is the planar
    // walker's y-down, so an arm must set it. When actuatedJoints is empty (the
    // default), every 1-DOF joint in the URDF is treated as actuated (a fully-actuated
    // arm): a throwaway probe model with no actuation is valid and lets us read
    // jointNames() (q/v order, "universe" excluded) to fill the list generically.
    // Pass an explicit list for an under-actuated fixed-base robot.
    inline RobotConfig fixedBaseArmConfig(const std::string &urdfPath,
                                          const Eigen::Vector3d &gravity = Eigen::Vector3d(0.0, 0.0,
                                                                                           -9.81),
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

    // TrajectoryProblem for a fixed-base arm reach. Builds the model, a single mode
    // (dynamics at every node; q,v pinned at both ends; optional joint/velocity/torque
    // box limits) under FullStateParam, and a linear-pose-interpolation initial guess.
    class FixedBaseReachProblem final : public TrajectoryProblem {
      public:
        FixedBaseReachProblem(const std::string &urdfPath, const Eigen::Vector3d &gravity,
                              Eigen::VectorXd q0, Eigen::VectorXd v0, Eigen::VectorXd qf,
                              Eigen::VectorXd vf, double T, int degree, bool enforceLimits) {
            if (degree < 4)
                throw std::invalid_argument(
                    "FixedBaseReachProblem: degree must be >= 4 (need interior DOF beyond the "
                    "two-point q/v boundary conditions)");
            if (T <= 0.0)
                throw std::invalid_argument("FixedBaseReachProblem: T must be > 0");

            model_ = std::make_shared<RobotModel>(
                RobotModel::fromUrdf(fixedBaseArmConfig(urdfPath, gravity)));
            param_ = std::make_shared<FullStateParam>();
            const RobotModel &m = *model_;
            const int nq = m.nq(), nv = m.nv(), nu = m.nu();
            auto checkSize = [](const Eigen::VectorXd &x, int n, const char *what) {
                if (static_cast<int>(x.size()) != n)
                    throw std::invalid_argument(std::string("FixedBaseReachProblem: ") + what +
                                                " has wrong size");
            };
            checkSize(q0, nq, "q0");
            checkSize(qf, nq, "qf");
            checkSize(v0, nv, "v0");
            checkSize(vf, nv, "vf");

            const ContactSet none{};
            const NodeDims d = m.nodeDims(none);
            using W = ConstraintAttachment::Where;

            ModeSpec ms;
            ms.degree = degree;
            ms.T = T;
            ms.contacts = none;
            // Manipulator dynamics at every node.
            ms.attachments.push_back({std::make_shared<RopeConstraint>(m, none), W::AllNodes});
            // Boundary conditions: pin q and v at t=0 (FirstNode) and t=T (LastNode).
            // q0/qf are copied (not moved): buildInitialGuess reads them below; v0/vf are
            // consumed only here, so they may be moved.
            ms.attachments.push_back(
                {std::make_shared<PinConstraint>(indexRange(0, nq), q0, "pin_q0"), W::FirstNode});
            ms.attachments.push_back(
                {std::make_shared<PinConstraint>(indexRange(d.offV(), nv), std::move(v0), "pin_v0"),
                 W::FirstNode});
            ms.attachments.push_back(
                {std::make_shared<PinConstraint>(indexRange(0, nq), qf, "pin_qf"), W::LastNode});
            ms.attachments.push_back(
                {std::make_shared<PinConstraint>(indexRange(d.offV(), nv), std::move(vf), "pin_vf"),
                 W::LastNode});
            // Physical limits from the URDF (opt-in). Position/velocity apply at interior
            // nodes only (the endpoints are already pinned to within-limits states);
            // torque applies everywhere. +/-inf sentinels auto-disable a side, so joints
            // the URDF leaves unbounded cost no rows.
            if (enforceLimits) {
                ms.attachments.push_back(
                    {std::make_shared<BoxConstraint>(BoxField::Q, indexRange(0, nq),
                                                     m.lowerPositionLimit(), m.upperPositionLimit(),
                                                     "box_q"),
                     W::Interior});
                const Eigen::VectorXd vlim = m.velocityLimit();
                ms.attachments.push_back({std::make_shared<BoxConstraint>(
                                              BoxField::V, indexRange(0, nv), -vlim, vlim, "box_v"),
                                          W::Interior});
                const Eigen::VectorXd ulim = m.actuatedEffortLimit();
                ms.attachments.push_back({std::make_shared<BoxConstraint>(
                                              BoxField::U, indexRange(0, nu), -ulim, ulim, "box_u"),
                                          W::AllNodes});
            }
            ms.costs = {std::make_shared<ControlEffortCost>()};
            spec_.modes = {std::move(ms)}; // single mode; spec_.resets stays empty

            buildInitialGuess(q0, qf);
        }

        std::shared_ptr<RobotModel> model() const override { return model_; }
        std::shared_ptr<const Parameterization> parameterization() const override { return param_; }
        const MultiModeSpec &spec() const override { return spec_; }
        const Eigen::VectorXd &initialGuess() const override { return y0_; }

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
        std::shared_ptr<const Parameterization> param_;
        MultiModeSpec spec_;
        Eigen::VectorXd y0_;
    };

} // namespace leap::examples
