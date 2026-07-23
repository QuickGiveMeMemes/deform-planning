// placeholder file
#pragma once
#include "leap/model/robot_config.hpp"
#include <string>
#include <vector>

namespace leap::examples {
    struct PinSpec {
        int vertex;
        std::string frameName;
    };

    inline std::vector<int> indexRange(int first, int count) {
        std::vector<int> v(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            v[static_cast<size_t>(i)] = first + i;
        }
        return v;
    }

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
}