#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <vector>

namespace nb = nanobind;

float BCELoss(const nb::DRef<Eigen::MatrixXf> &predictions,
              const nb::DRef<Eigen::MatrixXf> &targets) {
    std::cout << predictions << "\n";
    std::cout << targets << "\n";

    std::cout << predictions.rows() << "\n";
    int   N   = predictions.rows();
    int   M   = predictions.cols();
    float tot = static_cast<float>(0.0);
    for (size_t i = 0; i < N; ++i) {
        float sum = static_cast<float>(0.0);
        for (size_t j = 0; j < M; ++j) {
            sum += (targets.row(i)(j) * log(predictions.row(i)(j))) +
                   ((1 - targets.row(i)(j)) * log(1 - predictions.row(i)(j)));
        }
        sum *= (-1 / static_cast<float>(M));
        tot += sum;
    }
    /*
    nb::DRef<Eigen::MatrixXf> *p_predictions = predictions.data();
    nb::DRef<Eigen::MatrixXf> *p_targets     = targets.data();
    int                        N             = predictions.size();
    float                      sum           = static_cast<float>(0.0);
    int                        M;
    for (size_t i = 0; i < N; ++i) {
        nb::DRef<Eigen::MatrixXf> targets_vec     = *(p_targets + i);
        nb::DRef<Eigen::MatrixXf> predictions_vec = *(p_predictions + i);
        M                                         = targets_vec.size();
        /*
        for (size_t i = 0; i < M; ++i) {
            sum += (targets_vec(i) * log(predictions_vec(i))) +
                   ((1 - targets_vec(i)) * log(1 - predictions_vec(i)));
        }
        sum *= (-1 / static_cast<float>(M));

        std::cout << targets_vec;
        break;
    }*/

    return tot / N;
    // return sum;
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
