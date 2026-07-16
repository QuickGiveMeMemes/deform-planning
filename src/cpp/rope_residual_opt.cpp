// Fixed-base arm + rope residual solver (Kinova Gen3 7-DOF + DER rope).
// Rope dynamics (M a_r + grad E = 0) are hard equality constraints resolved by the
// ALM dual flow; the arm control is eliminated (u* = H a + C v + g + J_p^T f_p) and
// minimized as 1/2||u*||^2. Rope geometry/material come from a YAML config.
//
//   usage: kinova_rope alm [key=val ...]
//   urdf=<path>       robot URDF (default assets/kinova3_1arm.urdf).
//   rope=<path>       rope YAML (default assets/arm_rope_1pin.yaml).
//   pinframe=<name>   URDF frame the pinned rope vertex attaches to
//                     (default arm1_end_effector_link).
//   deg=<n>           collocation degree (default 16; Nn = deg+1; >=4).
//   T=<s>             horizon (default 3.0).
//   q0=/v0=/qf=/vf=   comma-separated arm boundary states (7 each).
//   gn=<0|1>          Gauss-Newton (1, default). gn=0 is REJECTED: RopeConstraint
//                     provides no addContractedHessian.
//   mumax/mu0/mugrow/tol/smax/dsseg/stall/backend  as in app_common.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../include/solvers/rope_residual.hpp"
#include <../../leap-discrete/refactored_codebase_v1/apps/app_common.hpp>
#include <leap/io/diagnostics.hpp>
#include <leap/io/internode.hpp>
#include <leap/solvers/coupled_flow.hpp>
#include <leap/solvers/linear_backend.hpp>

#include "../../include/coupling/rope_model.hpp"
#include <yaml-cpp/yaml.h>

using leap::app::argd;
using leap::app::args;
using leap::app::makeMomentum;
using leap::app::printAdaptive;
using leap::app::printFlowCounts;
using leap::app::wireAdaptive;
using leap::app::wireInterNodeReport;

namespace YAML {
    template <> struct convert<Eigen::VectorXd> {
        static bool decode(const Node &node, Eigen::VectorXd &vec) {
            if (!node.IsSequence()) {
                return false;
            }

            vec.resize(node.size());

            for (std::size_t i = 0; i < node.size(); ++i) {
                vec(i) = node[i].as<double>();
            }

            return true;
        }
    };
} // namespace YAML

namespace leap::examples {

    // Finite-difference check of RopeConstraint::jacobian at a given node state.
    // Central differences on the per-node decision vector z = [q; v; a; lam; u].
    // Prints the worst absolute/relative mismatch per field block.
    inline void fdCheckRope(const leap::RobotModel &m, const leap::examples::RopeConstraint &con,
                            const leap::NodeDims &d, const Eigen::VectorXd &z0,
                            const leap::ContactSet &cs, double eps = 1e-6) {
        auto pd = m.makeData();
        leap::ModelEvalCache mec;
        m.initCache(mec);
        const int R = con.rows(d), nP = d.nP();
        const int nqm = m.nq(), nvm = m.nv();

        auto residual = [&](const Eigen::VectorXd &z, Eigen::VectorXd &g) {
            Eigen::Map<const Eigen::VectorXd> q(z.data() + d.offQ(), d.nq);
            Eigen::Map<const Eigen::VectorXd> v(z.data() + d.offV(), d.nv);
            Eigen::Map<const Eigen::VectorXd> a(z.data() + d.offA(), d.na);
            Eigen::Map<const Eigen::VectorXd> lam(z.data() + d.offLam(), d.nlam);
            Eigen::Map<const Eigen::VectorXd> u(z.data() + d.offU(), d.nu);
            const leap::NodeQuantities P{q, v, a, lam, u};
            // SAME slice the assembler uses: pinocchio sees arm state only.
            m.evalNode(*pd, P.q.head(nqm), P.v.head(nvm), P.a.head(nvm), P.lam, cs,
                       con.modelNeeds(), mec);
            g.resize(R);
            con.value(P, d, mec, g);
        };

        // analytic J (needs its own eval to fill mec)
        Eigen::VectorXd g0(R);
        residual(z0, g0);
        Eigen::MatrixXd J(R, nP);
        {
            Eigen::Map<const Eigen::VectorXd> q(z0.data() + d.offQ(), d.nq);
            Eigen::Map<const Eigen::VectorXd> v(z0.data() + d.offV(), d.nv);
            Eigen::Map<const Eigen::VectorXd> a(z0.data() + d.offA(), d.na);
            Eigen::Map<const Eigen::VectorXd> lam(z0.data() + d.offLam(), d.nlam);
            Eigen::Map<const Eigen::VectorXd> u(z0.data() + d.offU(), d.nu);
            const leap::NodeQuantities P{q, v, a, lam, u};
            con.jacobian(P, d, mec, J);
        }

        Eigen::MatrixXd Jfd(R, nP);
        Eigen::VectorXd gp(R), gm(R), zp = z0, zm = z0;
        for (int j = 0; j < nP; ++j) {
            const double h = eps * std::max(1.0, std::abs(z0(j)));
            zp = z0;
            zm = z0;
            zp(j) += h;
            zm(j) -= h;
            residual(zp, gp);
            residual(zm, gm);
            Jfd.col(j) = (gp - gm) / (2.0 * h);
        }

        auto report = [&](const char *tag, int c0, int n) {
            if (n <= 0)
                return;
            const Eigen::MatrixXd A = J.middleCols(c0, n), B = Jfd.middleCols(c0, n);
            const double absErr = (A - B).cwiseAbs().maxCoeff();
            const double scale = std::max(1.0, B.cwiseAbs().maxCoeff());
            std::printf("  %-8s cols[%3d,%3d)  max|J-Jfd| = %.3e   rel = %.3e   max|Jfd| = %.3e\n",
                        tag, c0, c0 + n, absErr, absErr / scale, B.cwiseAbs().maxCoeff());
        };

        const int nfree3 = d.nq - nqm; // rope tail width
        std::printf("\n=== FD check: RopeConstraint::jacobian (R=%d, nP=%d, eps=%.1e) ===\n", R, nP,
                    eps);
        report("q_arm", d.offQ(), nqm);        // expect K_fp * J_p
        report("x_r", d.offQ() + nqm, nfree3); // expect K_ff
        report("v_arm", d.offV(), nvm);        // expect 0
        report("v_r", d.offV() + nvm, nfree3); // expect 0 (unless damping added)
        report("a_arm", d.offA(), nvm);        // expect 0
        report("a_r", d.offA() + nvm, nfree3); // expect M_ff
        const double tot = (J - Jfd).cwiseAbs().maxCoeff();
        std::printf("  OVERALL max|J - Jfd| = %.3e  %s\n", tot, tot < 1e-4 ? "[OK]" : "[MISMATCH]");
    }

    struct RopeYaml {
        RopeParams rp;
        Eigen::VectorXd damping; // 3N, per-DOF (UNUSED by the residual today -- see note)
    };

    inline RopeYaml loadRopeYaml(const std::string &path) {
        const YAML::Node y = YAML::LoadFile(path);
        RopeYaml out;
        RopeParams &rp = out.rp;

        // --- vertices (N x 3), also the rest configuration ---
        const YAML::Node &vs = y["vertices"];
        const int N = static_cast<int>(vs.size());
        if (N < 3)
            throw std::runtime_error("rope yaml: need >= 3 vertices");

        rp.rest_pos.resize(N, 3);
        for (int i = 0; i < N; ++i) {
            if (vs[i].size() != 3)
                throw std::runtime_error("rope yaml: vertex must be [x,y,z]");
            for (int j = 0; j < 3; ++j)
                rp.rest_pos(i, j) = vs[i][j].as<double>();
        }

        // --- mass: per-VERTEX in yaml -> per-DOF (3N) for mass(freeDof_) ---
        Eigen::VectorXd mv(N);
        const YAML::Node &mn = y["mass"];
        if (mn["mass_uniform"].as<bool>()) {
            mv.setConstant(mn["mass"].as<double>());
        } else {
            const YAML::Node &a = mn["mass_arr"];
            if (static_cast<int>(a.size()) != N)
                throw std::runtime_error("rope yaml: mass_arr length != vertex count");
            for (int i = 0; i < N; ++i)
                mv(i) = a[i].as<double>();
        }
        rp.mass.resize(3 * N);
        for (int i = 0; i < N; ++i)
            rp.mass.segment<3>(3 * i).setConstant(mv(i));

        // --- damping: same expansion; parsed but NOT consumed by the residual ---
        Eigen::VectorXd dv(N);
        const YAML::Node &dn = y["damping"];
        if (dn["damping_uniform"].as<bool>()) {
            dv.setConstant(dn["damping"].as<double>());
        } else {
            const YAML::Node &a = dn["damping_arr"];
            if (static_cast<int>(a.size()) != N)
                throw std::runtime_error("rope yaml: damping_arr length != vertex count");
            for (int i = 0; i < N; ++i)
                dv(i) = a[i].as<double>();
        }
        out.damping.resize(3 * N);
        for (int i = 0; i < N; ++i)
            out.damping.segment<3>(3 * i).setConstant(dv(i));

        // --- stretching (N-1 edges) ---
        //   rp.enable_stretch = y["stretching"]["enable"].as<bool>();
        rp.kstretch = Eigen::VectorXd::Constant(N - 1, y["stretching"]["stiffness"].as<double>());

        //   // --- bending (N-2 interior) ---
        //   rp.enable_bending = y["bending"]["enable_bending"].as<bool>();
        rp.kbend = Eigen::VectorXd::Constant(N - 2, y["bending"]["k_bend"].as<double>());

        // --- pinned vertices ---
        rp.pinned_idx.clear();
        for (const auto &p : y["control"]["pinned"]) {
            const int v = p.as<int>();
            if (v < 0 || v >= N)
                throw std::runtime_error("rope yaml: pinned index out of range");
            rp.pinned_idx.push_back(v);
        }
        if (rp.pinned_idx.empty())
            throw std::runtime_error("rope yaml: no pinned vertices");
        return out;
    }

} // namespace leap::examples

// Helpers for argparsing from cli
namespace {

    Eigen::VectorXd parseVec(const std::string &s, int n, const char *what) {
        std::vector<double> vals;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            bool blank = true;
            for (char c : tok)
                blank = blank && std::isspace(static_cast<unsigned char>(c));
            if (!blank)
                vals.push_back(std::stod(tok));
        }
        if (static_cast<int>(vals.size()) != n)
            throw std::runtime_error(std::string(what) + ": expected " + std::to_string(n) +
                                     " comma-separated values, got " + std::to_string(vals.size()));
        Eigen::VectorXd v(n);
        for (int i = 0; i < n; ++i)
            v(i) = vals[static_cast<size_t>(i)];
        return v;
    }

    Eigen::VectorXd vecArg(int argc, char **argv, const char *key, int n, const std::string &dflt) {
        const std::string s = args(argc, argv, key, dflt);
        if (s.empty())
            return Eigen::VectorXd::Zero(n);
        return parseVec(s, n, key);
    }

    // World position of monitor slot 0 at configuration q (v = a = 0), via the same
    // kMonitorPos FK path RopeConstraint reads at solve time.
    Eigen::Vector3d monitorPos0(const leap::RobotModel &m, const Eigen::VectorXd &q) {
        auto pd = m.makeData();
        leap::ModelEvalCache mec;
        m.initCache(mec);
        const Eigen::VectorXd z = Eigen::VectorXd::Zero(m.nv());
        m.evalNode(*pd, q, z, z, leap::kMonitorPos, mec);
        return mec.monPos.at(0);
    }

} // namespace

int main(int argc, char **argv) {
    const std::string solver = argc > 1 ? argv[1] : "alm";
    if (solver != "alm") {
        std::fprintf(stderr, "usage: %s alm [key=val ...]\n", argv[0]);
        return 2;
    }

    // Parse a ton of parameters
    const std::string problem = args(argc, argv, "problem", "src/data/problems/2arm1rope_v1.yaml");
    YAML::Node config = YAML::LoadFile(problem);

    const std::string urdf = config["urdf"].as<std::string>();
    const std::string ropeYaml =
        config["ropeYaml"].as<std::string>("src/data/rope/arm_rope_2pin.yaml");
    const std::vector<std::string> frames = config["pinFrames"].as<std::vector<std::string>>();

    const int deg = config["deg"].as<int>(16);
    const double T = config["T"].as<double>(3.0);

    const bool limits = config["limits"].as<bool>(false);
    const bool gn = config["gn"].as<bool>(true);
    const bool pfill = config["pfill"].as<bool>(true);

    const double muMax = config["muMax"].as<double>(1e7);
    const double mu0 = config["mu0"].as<double>(10.0);
    const double mugrow = config["mugrow"].as<double>(3.0);
    const double tol = config["tol"].as<double>(1e-6);
    const double dsSeg = config["dsSeg"].as<double>(5e-2);
    const double sMax = config["sMax"].as<double>(3.0);

    const int stallFlow = config["stallFlow"].as<int>(6);
    const double stallRel = config["stallRel"].as<double>(1e-2);
    const int interNodeReport = config["interNodeReport"].as<int>(16);

    const Eigen::Vector3d gravity(0.0, 0.0, -9.81);

    // RopeConstraint has no addContractedHessian -> exact Newton would silently drop
    // all rope curvature. Fail loudly rather than solve the wrong problem.
    if (!gn) {
        std::fprintf(stderr, "kinova_rope: gn=0 unsupported (no analytic rope Hessian)\n");
        return 2;
    }

    leap::SolverBackend backend = leap::SolverBackend::EigenLDLT;

    try {
        backend = leap::parseBackend(args(argc, argv, "backend", "accelu"));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }

    Eigen::VectorXd q0, v0, qf, vf;
    Eigen::MatrixXd x0_r, v0_r, xf_r;
    leap::examples::RopeYaml ry;
    std::vector<leap::examples::PinSpec> pins;
    int nq = 0, nv = 0, nRope = 0, nFree = 0;

    // Read rope configs
    try {
        ry = leap::examples::loadRopeYaml(ropeYaml);
        RopeParams &rp = ry.rp;
        nRope = static_cast<int>(rp.rest_pos.rows());
        nFree = nRope - static_cast<int>(rp.pinned_idx.size());

        if (nFree <= 0) {
            throw std::runtime_error("rope yaml: every vertex is pinned");
        }

        if (frames.size() != rp.pinned_idx.size()) {
            std::string e = "kinova_rope: pinframes count != yaml pinned count: " +
                            std::to_string(frames.size()) + ", " +
                            std::to_string(rp.pinned_idx.size());
            throw std::runtime_error(e);
        }

        for (size_t i = 0; i < frames.size(); ++i) {
            pins.push_back({rp.pinned_idx[i], frames[i]}); // yaml order == frame order
        }

        const leap::RobotModel probe =
            leap::RobotModel::fromUrdf(leap::examples::armConfig(urdf, frames, gravity));

        // Probe registers the SAME monitor frame the problem does, so slot 0 is
        // pinFrame in both and FK here matches FK at solve time.
        nq = probe.nq();
        nv = probe.nv();
        q0 = config["q0"].as<Eigen::VectorXd>(Eigen::VectorXd::Zero(nq));
        v0 = config["v0"].as<Eigen::VectorXd>(Eigen::VectorXd::Zero(nv));
        qf = config["qf"].as<Eigen::VectorXd>();
        vf = config["vf"].as<Eigen::VectorXd>(Eigen::VectorXd::Zero(nv));

        // FK/pins are verified consistent at q0, so the yaml shape is the initial state.
        x0_r = rp.rest_pos;
        v0_r = Eigen::MatrixXd::Zero(nRope, 3);

        // Terminal target: rigidly translate the shape by the pin's FK displacement
        // q0 -> qf. Preserves the elastic state (so the target is a reachable-ish rope
        // configuration) and keeps the pinned vertex coincident with FK(qf).
        const Eigen::Vector3d p0 = monitorPos0(probe, q0);
        const Eigen::Vector3d pf = monitorPos0(probe, qf);
        xf_r = x0_r;
        xf_r.rowwise() += (pf - p0).transpose();

        // Sanity: the pinned vertex must sit on its frame at q0 (else the elastic
        // residual starts huge and the pin constraint fights FK).
        const double pinErr = (x0_r.row(rp.pinned_idx[0]).transpose() - p0).norm();
        if (pinErr > 1e-6) {
            std::fprintf(stderr,
                         "kinova_rope: WARNING pinned vertex %d is %.3e m from "
                         "frame '%s' at q0\n",
                         rp.pinned_idx[0], pinErr, frames[rp.pinned_idx[0]].c_str());
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "kinova_rope: %s\n", e.what());
        return 2;
    }

    leap::examples::ContinuousRopeResidualProblem probObj(urdf, gravity, q0, v0, x0_r, v0_r, qf, vf,
                                                          xf_r, ry.rp, pins, T, deg);
    leap::CompiledProblem prob = probObj.compile();

    const leap::SegmentLayout &sl = prob.layout().seg[0];
    const Eigen::VectorXd y0 = probObj.initialGuess();
    const int k = sl.Nn / 2; // an interior node

    leap::TrajectoryView V(y0.data(), sl);
    Eigen::VectorXd z(sl.dims.nP());
    V.node(k).toVector(sl.dims, z);

    leap::examples::RopeConstraint con(*probObj.model(), ry.rp, pins);
    fdCheckRope(*probObj.model(), con, sl.dims, z, leap::ContactSet{}, 1e-6);

    std::printf("[kinova_rope|alm] urdf=%s  rope=%s\n"
                "  arm: nq=%d nv=%d nu=0 (control eliminated)   decision nq=%d\n"
                "  rope: n=%d free=%d pin=v%d@  ks=%.3g  bend=%s(kb=%.3g)  damping=%s\n"
                "  deg=%d T=%.2f  gn=ON  mumax=%.3g  backend=%s\n",
                urdf.c_str(), ropeYaml.c_str(), nq, nv, nq + 3 * nFree, nRope, nFree,
                ry.rp.pinned_idx[0], ry.rp.kstretch(0), "ON",
                ry.rp.kbend.size() ? ry.rp.kbend(0) : 0.0, "UNMODELED", deg, T, muMax,
                leap::backendName(backend));
    std::printf("Pinned frames: ");
    for (const std::string &f : frames) {
        std::printf("%s ", f.c_str());
    }
    std::printf("\n");

    leap::KktAssembler asmb(prob, probObj.formulation());
    asmb.setParallelFill(pfill);
    asmb.setGaussNewton(true);

    leap::MuPolicyConfig muCfg;
    muCfg.muMax = muMax;
    muCfg.mu0 = mu0;
    muCfg.rho = mugrow;
    leap::PenaltyManager pen(prob, /*perTerm=*/true, muCfg);

    leap::FlowConfig cfg;
    cfg.theta = 1.0;
    cfg.dsSeg = dsSeg;
    cfg.sMax = sMax;
    cfg.tol = tol;
    cfg.stallWindows = stallFlow;
    cfg.stallRel = stallRel;
    cfg.keepBest = true;
    cfg.backend = backend;

    wireAdaptive(cfg, argc, argv);
    cfg.header = "[kinova_rope|alm] single-phase coupled Nesterov flow (MuStall cap " +
                 std::to_string(muMax) + ", s_max=" + std::to_string(sMax) +
                 ", backend=" + leap::backendName(backend) + ")";
    cfg.onReport = leap::io::printViolationsByKind;

    wireInterNodeReport(cfg, interNodeReport);

    Eigen::VectorXd y = probObj.initialGuess();

    const auto t0 = std::chrono::steady_clock::now();

    leap::CoupledFlowDriver driver(asmb, pen, makeMomentum(argc, argv), cfg);
    leap::FlowReport rep = driver.run(y);

    const char *tag = "[kinova_rope|alm]";

    printFlowCounts(tag, rep);
    printAdaptive(tag, rep);

    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("\n%s TOTAL %.1f s  (converged=%d, max|c_eff|=%.3e at s=%.3f)\n", tag, wall,
                static_cast<int>(rep.converged), rep.maxViol, rep.s);

    // Judging: full space, zero duals (mu-free feasibility gate).
    leap::EvalCache cache = asmb.makeCache();

    {
        Eigen::VectorXd yJ = y;
        yJ.tail(prob.layout().md).setZero();
        asmb.evaluate(yJ, pen.muRow(), {}, cache);
    }

    leap::io::printGroupViolations(prob, cache);
    leap::io::printViolationsByKind(prob, cache);

    std::printf("  gate (zero-dual) max|c| = %.3e   ||dZ/ds|| = %.3e\n", cache.maxViol,
                cache.F.head(prob.layout().np).norm());

    leap::io::printInterNodeViolations(prob, y);

    leap::io::writeSolutionViz(prob, cache, y, "sol_kinova_rope_alm", urdf);

    std::printf("solution written to sol_kinova_rope_alm_mode0.csv (+ _meta.json, "
                "_cons_mode0.csv)\n");

    return 0;
}