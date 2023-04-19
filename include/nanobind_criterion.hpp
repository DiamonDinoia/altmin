#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <vector>

namespace nb = nanobind;

float BCELoss(const nb::DRef<Eigen::MatrixXd> &predictions,
              const nb::DRef<Eigen::MatrixXd> &targets) {
    int    N   = predictions.rows();
    int    M   = predictions.cols();
    double tot = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < M; ++j) {
            sum += (targets.row(i)(j) * std::log(predictions.row(i)(j))) +
                   ((1.0 - targets.row(i)(j)) *
                    std::log(1.0 - predictions.row(i)(j)));
        }
        sum *= (-1.0 / static_cast<double>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}

float MSELoss(const std::vector<float> &predictions,
              const std::vector<float> &targets) {
    int   N   = predictions.size();
    float sum = static_cast<float>(0.0);
    for (size_t i = 0; i < N; ++i) {
        sum += (targets[i] - predictions[i]) * (targets[i] - predictions[i]);
    }
    sum *= (1 / static_cast<float>(N));
    return sum;
}
