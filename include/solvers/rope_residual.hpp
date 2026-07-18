#pragma once

// TODO split this into like 5 files

// Residual-based arm-rope optimizer. The rope dynamics residual is enforced via the
// dual ALM flow, while the arm control is minimized via least-squares
//
// Ha+C=u -> least squares 1/2 || Ha+C ||^2 in cost

#include <algorithm>
#include <memory>
#include <pinocchio/multibody/fwd.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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
#include <leap/spectral/spectral_ops.hpp>
#include <leap/transcription/compiled_problem.hpp>
#include <leap/transcription/layout.hpp>
#include <leap/transcription/parameterization.hpp>

#include <leap/core/node_dims.hpp>
#include <leap/core/node_quantities.hpp>

#include "../coupling/rope_model.hpp"

namespace leap::examples {

    namespace pin = pinocchio;

    struct PinSpec {
        int vertex;
        std::string frameName;
    };

    // Pinned vertices are not decision variables in this structure
    //
    // Rope residual dual constraint enforcing 0 = Ma + grad (sum of elastic energies)
    struct RopeConstraint final : public ConstraintBase {

        // TODO add checking of coincidence between pinned nodes and corresponding rope vert so the
        // physics do not blow up Monitor frames assumed to be specified already
        RopeConstraint(const RobotModel &m, const RopeParams &rp, std::vector<PinSpec> pins)
            : rope_(rp), nq_arm_(m.nq()), nv_arm_(m.nv()), nvert_(rp.rest_pos.rows()),
              name_("rope") {

            // Pinned are not decision variables, they are queried from the robot model
            // Sort by vertex so pinVert_/pinSlot_ are ascending == split()'s pinned DOF order
            std::sort(pins.begin(), pins.end(),
                      [](const PinSpec &a, const PinSpec &b) { return a.vertex < b.vertex; });

            std::unordered_map<std::string, int> slotOf;
            for (int i = 0; i < m.nMonitors(); ++i) {
                slotOf[m.monitorName(i)] = i;
            }

            is_pin_.assign(nvert_, 0);

            for (const PinSpec &p : pins) {
                auto it = slotOf.find(p.frameName);
                if (it == slotOf.end()) {
                    throw std::runtime_error("RopeConstraint: pin frame '" + p.frameName +
                                             "' is not a registered monitor frame");
                }

                is_pin_[p.vertex] = 1;
                pinVert_.push_back(p.vertex);   // rope-vertex index
                pinSlot_.push_back(it->second); // monitor slot (indexes monPos/monJ)
            }

            npin_ = static_cast<int>(pins.size());
            nfree_ = nvert_ - npin_;
            for (int v = 0; v < nvert_; ++v) {
                if (!is_pin_[v]) {
                    for (int j = 0; j < 3; ++j) {
                        freeDof_.push_back(3 * v + j);
                    }
                }
            }

            mass_free_ = rp.mass(freeDof_);

            grav_free_.resize(3 * nfree_);
            for (int i = 0; i < nfree_; ++i) {
                grav_free_.segment<3>(3 * i) = Eigen::Vector3d(0, 0, -9.8); // currently hardcoded
            }
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

            g = mass_free_.asDiagonal() * (P.a.tail(3 * nfree_) - grav_free_) + grad_f;
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
            J.middleCols(d.offA() + nv_arm_, 3 * nfree_) =
                mass_free_.asDiagonal(); // dtau/da = mass matrix
        }

        // No hessian

        Eigen::MatrixXd buildR(const NodeQuantities &P, const ModelEvalCache &c) const {

            Eigen::MatrixXd R(nvert_, 3);
            auto xr = P.q.tail(3 * nfree_);
            int fi = 0, pi = 0; // free, pinned

            for (int v = 0; v < nvert_; ++v) {
                if (is_pin_[v]) {
                    R.row(v) = c.monPos[pinSlot_[pi++]].transpose();
                } else {
                    R.row(v) = xr.segment(3 * fi++, 3).transpose();
                }
            }
            return R;
        }

        DynamicRopeModel rope_;
        std::vector<char> is_pin_;
        std::vector<int> pinVert_, pinSlot_, freeDof_;
        Eigen::VectorXd mass_free_, grav_free_;
        int nq_arm_, nv_arm_, nvert_, nfree_ = 0, npin_ = 0;
        std::string name_;
    };

    // TODO: a lot of code repetition between this and the rope residual, could structure in a
    // cleaner form
    struct ArmControlCost final : public CostBase {

        ArmControlCost(const RobotModel &m, const RopeParams &rp, std::vector<PinSpec> pins)
            : rope_(rp), nq_arm_(m.nq()), nv_arm_(m.nv()), nvert_(rp.rest_pos.rows()),
              name_("arm") {

            // Pinned are not decision variables, they are queried from the robot model
            // Sort by vertex so pinVert_/pinSlot_ are ascending == split()'s pinned DOF order
            std::sort(pins.begin(), pins.end(),
                      [](const PinSpec &a, const PinSpec &b) { return a.vertex < b.vertex; });

            std::unordered_map<std::string, int> slotOf;
            for (int i = 0; i < m.nMonitors(); ++i)
                slotOf[m.monitorName(i)] = i;

            is_pin_.assign(nvert_, 0);

            for (const PinSpec &p : pins) {
                auto it = slotOf.find(p.frameName);
                if (it == slotOf.end()) {
                    throw std::runtime_error("RopeConstraint: pin frame '" + p.frameName +
                                             "' is not a registered monitor frame");
                }

                is_pin_[p.vertex] = 1;
                pinVert_.push_back(p.vertex);   // rope-vertex index
                pinSlot_.push_back(it->second); // monitor slot (indexes monPos/monJ)
            }

            npin_ = static_cast<int>(pins.size());
            nfree_ = nvert_ - npin_;

            // TODO can delete probably - freeDof_ is unused
            for (int v = 0; v < nvert_; ++v) {
                if (!is_pin_[v])
                    for (int j = 0; j < 3; ++j) {
                        freeDof_.push_back(3 * v + j);
                    }
            }
        }

        unsigned modelNeeds() const override { return kRnea | kRneaDerivs | kMonitorPos; }

        double value(const NodeQuantities &P, const NodeDims &,
                     const ModelEvalCache &c) const override {

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
            for (int i = 0; i < npin_; ++i) {
                Jp.middleRows(3 * i, 3) = c.monJ[pinSlot_[i]];
            }

            // Least squares 1/2 || Ha + C + Jp^T(reaction) ||
            return (c.tau + Jp.transpose() * grad_p).squaredNorm() / 2.0 * 1e-4;
        }

        void gradient(const NodeQuantities &P, const NodeDims &d, const ModelEvalCache &c,
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
            g_P.segment(d.offQ(), nq_arm_) =
                c.dtau_dq.transpose() * u + Jp.transpose() * Kpp * Jp * u;
            g_P.segment(d.offQ() + nq_arm_, nfree_ * 3) = Kfp * Jp * u;
            g_P.segment(d.offV(), nv_arm_) = c.dtau_dv.transpose() * u;
            g_P.segment(d.offA(), nv_arm_) = c.M * u;

            g_P *= 1e-4;
        }

        Eigen::MatrixXd buildR(const NodeQuantities &P, const ModelEvalCache &c) const {

            Eigen::MatrixXd R(nvert_, 3);
            auto xr = P.q.tail(3 * nfree_);
            int fi = 0, pi = 0; // free, pinned

            for (int v = 0; v < nvert_; ++v) {
                if (is_pin_[v]) {
                    R.row(v) = c.monPos[pinSlot_[pi++]].transpose();
                } else {
                    R.row(v) = xr.segment(3 * fi++, 3).transpose();
                }
            }
            return R;
        }

        DynamicRopeModel rope_;
        std::vector<char> is_pin_;
        std::vector<int> pinVert_, pinSlot_, freeDof_;
        Eigen::VectorXd mass_pinned_;
        int nq_arm_, nv_arm_, nvert_, nfree_ = 0, npin_ = 0;
        std::string name_;
    };

    // Soft least-squares boundary cost
    // struct BoundaryCost final : public CostBase {

    //     // i0 is based on total indexing (NodeQuantities::atP())
    //     BoundaryCost(int i0, Eigen::VectorXd target, double k)
    //         : i0_(i0), target_(std::move(target)),
    //             len_(static_cast<int>(target_.size())), k_(k) {}

    //     unsigned modelNeeds() const override { return 0u; }

    //     double value(const NodeQuantities& P, const NodeDims& d, const ModelEvalCache&) const
    //     override {
    //         double s = 0.0;
    //         for (int j = 0; j < len_; ++j) {
    //             const double e = P.atP(d, i0_ + j) - target_(j);
    //             s += e * e;
    //         }
    //         return 0.5 * k_ * s;
    //     }

    //     void gradient(const NodeQuantities& P, const NodeDims& d, const ModelEvalCache&,
    //                     Eigen::Ref<Eigen::VectorXd> g_P) const override {
    //         g_P.setZero();
    //         for (int j = 0; j < len_; ++j)
    //             g_P(i0_ + j) = k_ * (P.atP(d, i0_ + j) - target_(j));
    //     }

    //     Eigen::VectorXd target_;
    //     int i0_, len_;
    //     double k_;
    // };

    // Contiguous index list [first, first+count) into a per-node [q;v;a;lam;u] vector.
    inline std::vector<int> indexRange(int first, int count) {
        std::vector<int> v(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            v[static_cast<size_t>(i)] = first + i;
        }
        return v;
    }

    // RobotConfig for a FIXED-BASE robot: no contacts, no monitor frames. gravity
    // defaults to z-up (0,0,-9.81) -- the RobotConfig struct default is the planar
    // walker's y-down, so an arm must set it. When actuatedJoints is empty (the
    // default), every 1-DOF joint in the URDF is treated as actuated (a fully-actuated
    // arm): a throwaway probe model with no actuation is valid and lets us read
    // jointNames() (q/v order, "universe" excluded) to fill the list generically.
    // Pass an explicit list for an under-actuated fixed-base robot.
    inline RobotConfig
    armConfig(const std::string &urdfPath, const std::vector<std::string> &pinnedFrames,
              const Eigen::Vector3d &gravity = Eigen::Vector3d(0.0, 0.0, -9.81)) {

        RobotConfig cfg;
        cfg.urdfPath = urdfPath;
        cfg.gravity = gravity;

        // const RobotModel m0 = RobotModel::fromUrdf(probe);
        cfg.actuatedJoints = {};

        cfg.monitorFrames = pinnedFrames;
        return cfg;
    }

    class ContinuousRopeResidualProblem final : public TrajectoryProblem {
      public:
        ContinuousRopeResidualProblem(
            const std::string &urdfPath, const Eigen::Vector3d &gravity, Eigen::VectorXd q0_arm,
            Eigen::VectorXd v0_arm, Eigen::MatrixXd x0_r,
            Eigen::MatrixXd v0_r, // Full arm config (pinned will be removed)
            Eigen::VectorXd qf_arm, Eigen::VectorXd vf_arm,
            Eigen::MatrixXd xf_r, // Eigen::MatrixXd vf_r,
            RopeParams &rin, std::vector<PinSpec> pins, double T, int degree)
            : rp(rin) {

            if (degree < 4) {
                throw std::invalid_argument("Degree must be >= 4 (need interior DOF beyond the "
                                            "two-point q/v boundary conditions)");
            }

            if (T <= 0.0) {
                throw std::invalid_argument("T must be > 0");
            }

            const int n_rope = rin.rest_pos.rows();
            const int N3 = n_rope * 3; // xyz

            std::vector<char> is_pin(n_rope, 0);
            for (const PinSpec &p : pins) {
                is_pin[p.vertex] = 1;
            }

            std::vector<int> freeVerts;

            for (int v = 0; v < n_rope; ++v) {
                if (!is_pin[v]) {
                    freeVerts.push_back(v);
                }
            }

            const int n_free = static_cast<int>(freeVerts.size());
            const int N3f = 3 * n_free;

            std::vector<std::string> pinned;
            pinned.reserve(rin.pinned_idx.size());

            for (auto &p : pins) {
                pinned.push_back(p.frameName);
            }

            model_ = std::make_shared<RobotModel>(
                RobotModel::fromUrdf(armConfig(urdfPath, pinned, gravity)));
            param_ = std::make_shared<FullStateParam>();
            const RobotModel &m = *model_;
            const int nq_arm = m.nq(), nv_arm = m.nv();

            const ContactSet none{};

            auto freeFlat = [&](const Eigen::MatrixXd &X) -> Eigen::VectorXd {
                const Eigen::MatrixXd Xf = X(freeVerts, Eigen::all); // n_free x 3
                return Xf.transpose().reshaped(); // [x0,y0,z0,x1,...] == DOF 3*v+j
            };

            const Eigen::VectorXd x0f = freeFlat(x0_r), v0f = freeFlat(v0_r), xff = freeFlat(xf_r);
            // auto vf_r_flat = vf_r.reshaped<Eigen::RowMajor>();

            // nlam, nu should be 0
            NodeDims d;
            d.nq = m.nq() + N3f;
            d.nv = m.nv() + N3f;
            d.na = m.nv() + N3f;

            using W = ConstraintAttachment::Where;

            ModeSpec ms;
            ms.extraDims = N3f;
            ms.degree = degree;
            ms.T = T;
            ms.contacts = none;

            // Node-enforced rope dynamics
            ms.attachments.push_back({std::make_shared<RopeConstraint>(m, rp, pins), W::AllNodes});

            // Boundary conditions
            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
                                                                      q0_arm, "pin_q0_arm"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      v0_arm, "pin_v0_arm"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(
                                          indexRange(d.offQ() + nq_arm, N3f), x0f, "pin_x0_rope"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(
                                          indexRange(d.offV() + nv_arm, N3f), v0f, "pin_v0_rope"),
                                      W::FirstNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offQ(), nq_arm),
                                                                      qf_arm, "pin_qf_arm"),
                                      W::LastNode});

            ms.attachments.push_back({std::make_shared<PinConstraint>(indexRange(d.offV(), nv_arm),
                                                                      vf_arm, "pin_vf_arm"),
                                      W::LastNode});

            // Final rope conditions are soft costs instead of hard constraints
            // not anymore

            // Only enforcing position, at least until LEAP supports location-specific cost
            // ms.attachments.push_back(
            //     {std::make_shared<PinConstraint>(indexRange(d.offQ() + nq_arm, N3f), xff,
            //     "pin_xf_rope"), W::LastNode}
            // );

            ms.costs.push_back(std::make_shared<ArmControlCost>(m, rp, pins));

            spec_.modes = {std::move(ms)}; // single mode; spec_.resets stays empty

            Eigen::VectorXd q0(d.nq), qf(d.nq);
            q0 << q0_arm, x0f;
            qf << qf_arm, xff;

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
        RopeParams rp;

        std::shared_ptr<const Parameterization> param_;
        MultiModeSpec spec_;
        Eigen::VectorXd y0_;
    };

} // namespace leap::examples
