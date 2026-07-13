#pragma once

// TODO split this into like 5 files

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

#include <leap/examples/trajectory_problem.hpp>
#include <leap/model/robot_config.hpp>
#include <leap/model/robot_model.hpp>
#include <leap/problem/constraint_base.hpp>
#include <leap/problem/constraints.hpp>
#include <leap/problem/costs.hpp>
#include <leap/problem/mode_spec.hpp>
#include <leap/spectral/spectral_ops.hpp>
#include <leap/transcription/compiled_problem.hpp>
#include <leap/transcription/layout.hpp>
#include <leap/transcription/parameterization.hpp>

#include <leap/core/node_dims.hpp>
#include <leap/core/node_quantities.hpp>

#include "../coupling/rope_model.hpp"

namespace leap::examples {
    
    namespace pin = pinocchio;

    struct PinSpec { int vertex; std::string frameName; };

    // Pinned vertices are not decision variables in this structure
    //
    // Rope residual dual constraint enforcing 0 = Ma + grad (sum of elastic energies)
    struct RopeConstraint final : public ConstraintBase {

        // TODO add checking of coincidence between pinned nodes and corresponding rope vert so the physics do not blow up
        // Monitor frames assumed to be specified already
        RopeConstraint(const RobotModel& m, const RopeParams& rp, std::vector<PinSpec> pins)
            :   rope_(rp), nq_arm_(m.nq()), nv_arm_(m.nv()), nvert_(rp.rest_pos.rows()), name_("rope") {
            
            // Pinned are not decision variables, they are queried from the robot model
            // Sort by vertex so pinVert_/pinSlot_ are ascending == split()'s pinned DOF order
            std::sort(pins.begin(), pins.end(),
                    [](const PinSpec& a, const PinSpec& b) { return a.vertex < b.vertex; });

            std::unordered_map<std::string, int> slotOf;
            for (int i = 0; i < m.nMonitors(); ++i) 
                slotOf[m.monitorName(i)] = i;

            is_pin_.assign(nvert_, 0);
            
            for (const PinSpec& p : pins) {
                auto it = slotOf.find(p.frameName);
                if (it == slotOf.end())
                    throw std::runtime_error("RopeConstraint: pin frame '" + p.frameName +
                                            "' is not a registered monitor frame");

                is_pin_[p.vertex] = 1;
                pinVert_.push_back(p.vertex);    // rope-vertex index
                pinSlot_.push_back(it->second);  // monitor slot (indexes monPos/monJ)
            }

            npin_  = static_cast<int>(pins.size());
            nfree_ = nvert_ - npin_;
            for (int v = 0; v < nvert_; ++v)
                if (!is_pin_[v]) for (int j = 0; j < 3; ++j) 
                    freeDof_.push_back(3 * v + j);
            
            mass_free_ = rp.mass(freeDof_);
        }

        const std::string &name() const override { return name_; }
        int rows(const NodeDims &) const override { return 3 * nfree_; }
        Sense sense() const override { return Sense::Equality; }
        unsigned modelNeeds() const override { return kMonitorPos; }

        void value(const NodeQuantities &P, const NodeDims &, const ModelEvalCache &c,
                   Eigen::Ref<Eigen::VectorXd> g) const override {
            
            // TODO: currently nodequantities -> r is flat->non-flat->flat, can make more efficient
            const auto R = buildR(P, c);

            auto [grad_f, grad_p] = rope_.elastic_grad(R, 0);
            const auto [grad_f_b, grad_p_b] = rope_.elastic_grad(R, 1);
            grad_f += grad_f_b;
            grad_p += grad_p_b;

            g = mass_free_.asDiagonal() * P.a.tail(3 * nfree_) + grad_f;
        }

        void jacobian(const NodeQuantities &P, const NodeDims &d, const ModelEvalCache &c,
                      MatRef J) const override {
            
            // J is of size (n_r_free, 3 * n)

            const auto R = buildR(P, c);

            auto [Kff, Kfp, Kpp] = rope_.elastic_hess(R, 0);
            const auto [Kff_b, Kfp_b, Kpp_b] = rope_.elastic_hess(R, 1);
            
            Kff += Kff_b;
            Kfp += Kfp_b;

            Eigen::MatrixXd Jp(3 * npin_, nv_arm_);
            for (int i = 0; i < npin_; ++i) 
                Jp.middleRows(3 * i, 3) = c.monJ[pinSlot_[i]];
            
            J.setZero();
            J.middleCols(d.offQ(), nq_arm_) = Kfp * Jp; // == dtau_dq when cs empty
            J.middleCols(d.offQ() + nq_arm_, 3 * nfree_) = Kff;
            J.middleCols(d.offA() + nv_arm_, 3 * nfree_) = mass_free_.asDiagonal(); // dtau/da = mass matrix
            
        }

        // No hessian

        Eigen::MatrixXd buildR(const NodeQuantities& P, const ModelEvalCache& c) const {
            
            Eigen::MatrixXd R(nvert_, 3);
            auto xr = P.q.tail(3 * nfree_);
            int fi = 0, pi = 0; // free, pinned

            for (int v = 0; v < nvert_; ++v) {
                if (is_pin_[v]) 
                    R.row(v) = c.monPos[pinSlot_[pi++]].transpose();
                else            
                    R.row(v) = xr.segment(3 * fi++, 3).transpose();
            }
            return R;
        }

        DynamicRopeModel rope_;
        std::vector<char> is_pin_;
        std::vector<int>  pinVert_, pinSlot_, freeDof_;
        Eigen::VectorXd   mass_free_;
        int nq_arm_, nv_arm_, nvert_, nfree_ = 0, npin_ = 0;
        std::string name_;

    };

    // TODO: a lot of code repetition between this and the rope residual, could structure in a cleaner form
    struct ArmControlCost final : public CostBase {

        ArmControlCost(const RobotModel& m, const RopeParams& rp, std::vector<PinSpec> pins)
            :   rope_(rp), nq_arm_(m.nq()), nv_arm_(m.nv()), nvert_(rp.rest_pos.rows()), name_("arm") {
            
            // Pinned are not decision variables, they are queried from the robot model
            // Sort by vertex so pinVert_/pinSlot_ are ascending == split()'s pinned DOF order
            std::sort(pins.begin(), pins.end(),
                    [](const PinSpec& a, const PinSpec& b) { return a.vertex < b.vertex; });

            std::unordered_map<std::string, int> slotOf;
            for (int i = 0; i < m.nMonitors(); ++i) 
                slotOf[m.monitorName(i)] = i;

            is_pin_.assign(nvert_, 0);
            
            for (const PinSpec& p : pins) {
                auto it = slotOf.find(p.frameName);
                if (it == slotOf.end())
                    throw std::runtime_error("RopeConstraint: pin frame '" + p.frameName +
                                            "' is not a registered monitor frame");

                is_pin_[p.vertex] = 1;
                pinVert_.push_back(p.vertex);    // rope-vertex index
                pinSlot_.push_back(it->second);  // monitor slot (indexes monPos/monJ)
            }

            npin_  = static_cast<int>(pins.size());
            nfree_ = nvert_ - npin_;

            // TODO can delete probably
            for (int v = 0; v < nvert_; ++v)
                if (!is_pin_[v]) 
                    for (int j = 0; j < 3; ++j) 
                        freeDof_.push_back(3 * v + j);
            
            mass_pinned_ = rp.mass(pinVert_);
        }

        unsigned modelNeeds() const override {  return kRnea | kRneaDerivs | kMonitorPos; }

        double value(const NodeQuantities& P, const NodeDims&, const ModelEvalCache& c) const override {

            // "residual" force of pinned nodes is the reaction force
            // This does not include the inertial force of the pinned node as it is
            // quite a significant calculation chain for not much gain
            // i.e. F_react = J_p^T[Ma+grad (elastic force)]_pinned, but
            // Ma is discarded
            //
            // TODO, should be able to disable reaction if necessary

            const auto R = buildR(P, c);

            auto [grad_f, grad_p] = rope_.elastic_grad(R, 0);
            const auto [grad_f_b, grad_p_b] = rope_.elastic_grad(R, 1);
            grad_p += grad_p_b; // Elastic reaction

            Eigen::MatrixXd Jp(3 * npin_, nv_arm_);
            for (int i = 0; i < npin_; ++i) 
                Jp.middleRows(3 * i, 3) = c.monJ[pinSlot_[i]];

            // Least squares 1/2 || Ha + C + Jp^T(reaction) ||
            return (c.tau + Jp.transpose() * grad_p).squaredNorm() / 2.0;
        }

        void gradient(const NodeQuantities& P, const NodeDims& d, const ModelEvalCache& c,
                        Eigen::Ref<Eigen::VectorXd> g_P) const override {
            
            const auto R = buildR(P, c);
            
            auto [Kff, Kfp, Kpp] = rope_.elastic_hess(R, 0);
            const auto [Kff_b, Kfp_b, Kpp_b] = rope_.elastic_hess(R, 1);
            
            Kff += Kff_b;
            Kfp += Kfp_b;
            Kpp += Kpp_b;
            
            // Some more code repetition
            auto [grad_f, grad_p] = rope_.elastic_grad(R, 0);
            const auto [grad_f_b, grad_p_b] = rope_.elastic_grad(R, 1);
            grad_p += grad_p_b; // Elastic reaction
                    
            Eigen::MatrixXd Jp(3 * npin_, nv_arm_);
            for (int i = 0; i < npin_; ++i) 
                Jp.middleRows(3 * i, 3) = c.monJ[pinSlot_[i]];

            Eigen::VectorXd u = c.tau + Jp.transpose() * grad_p;
            
            g_P.setZero();
            g_P.segment(d.offQ(), nq_arm_) = c.dtau_dq.transpose() * u + Jp.transpose() * Kpp * Jp * u;
            g_P.segment(d.offQ() + nq_arm_, nfree_ * 3) = Kfp * Jp * u;
            g_P.segment(d.offV(), nv_arm_) = c.dtau_dv.transpose() * u;
            g_P.segment(d.offA(), nv_arm_) = c.M * u;
        }


        Eigen::MatrixXd buildR(const NodeQuantities& P, const ModelEvalCache& c) const {
            
            Eigen::MatrixXd R(nvert_, 3);
            auto xr = P.q.tail(3 * nfree_);
            int fi = 0, pi = 0; // free, pinned

            for (int v = 0; v < nvert_; ++v) {
                if (is_pin_[v]) 
                    R.row(v) = c.monPos[pinSlot_[pi++]].transpose();
                else            
                    R.row(v) = xr.segment(3 * fi++, 3).transpose();
            }
            return R;
        }

        DynamicRopeModel rope_;
        std::vector<char> is_pin_;
        std::vector<int>  pinVert_, pinSlot_, freeDof_;
        Eigen::VectorXd   mass_pinned_;
        int nq_arm_, nv_arm_, nvert_, nfree_ = 0, npin_ = 0;
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
