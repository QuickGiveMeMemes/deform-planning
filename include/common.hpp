// placeholder file
#pragma once
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
}