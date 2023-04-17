#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <vector>

namespace nb = nanobind;

float BCELoss(const std::vector<float> &predictions,
              const std::vector<float> &targets) {
    int   N   = predictions.size();
    float sum = static_cast<float>(0.0);
    for (size_t i = 0; i < N; ++i) {
        sum += (targets[i] * log(predictions[i])) +
               ((1 - targets[i]) * log(1 - predictions[i]));
    }
    sum *= (-1 / static_cast<float>(N));
    return sum;
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

NB_MODULE(nanobind_criterion, m) {
    m.def("BCELoss", &BCELoss);
    m.def("MSELoss", &MSELoss);
    m.doc() = "Implementation of criterion in c++";
}
