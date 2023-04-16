#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace Eigen;

MatrixXf lin(const nb::DRef<Eigen::MatrixXf> &input,
             const nb::DRef<Eigen::MatrixXf> &weight,
             const nb::DRef<Eigen::MatrixXf> &bias) {
    MatrixXf result = (input * weight.transpose()).rowwise() + bias.row(0);
    if (result.rows() == 1) { result = ((input * weight.transpose()) + bias); }
    return result;
}

Eigen::MatrixXf ReLU(const nb::DRef<Eigen::MatrixXf> &input) {
    return input.unaryExpr(
        [](float x) { return std::max(x, static_cast<float>(0.0)); });
}

Eigen::MatrixXf sigmoid(const nb::DRef<Eigen::MatrixXf> &input) {
    return input.unaryExpr([](float x) {
        return static_cast<float>(1.0) / (static_cast<float>(1.0) + exp(-x));
    });
}

NB_MODULE(nanobind_layers, m) {
    m.def("lin", &lin);
    m.def("ReLU", &ReLU);
    m.def("sigmoid", &sigmoid);
    m.doc() = "nn layers in c++";
}
