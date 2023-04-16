#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include "nanobind_layers.hpp"

namespace nb = nanobind;

Eigen::MatrixXf lin(const nb::DRef<Eigen::MatrixXf> &input,
                    const nb::DRef<Eigen::MatrixXf> &weight,
                    const nb::DRef<Eigen::MatrixXf> &bias) {
    Eigen::MatrixXf result =
        (input * weight.transpose()).rowwise() + bias.row(0);
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

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const Eigen::EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// use int map to avoid strings
// 0 = linear
// 1 = ReLU
// 2 = sigmoid
std::tuple<Eigen::MatrixXf, std::vector<Eigen::MatrixXf>> get_codes(
    const std::vector<int> &model_mods, const int num_codes,
    const nb::DRef<Eigen::MatrixXf>       &inputs,
    std::vector<nb::DRef<Eigen::MatrixXf>> tmp) {
    Eigen::MatrixXf              result = inputs;
    std::vector<Eigen::MatrixXf> codes;
    int                          codes_index = 0;
    int                          x           = 0;
    nb::DRef<Eigen::MatrixXf>   *p_w         = tmp.data();

    for (auto const &i : model_mods) {
        if (i == 0) {
            result = lin(result, *(p_w + x), *(p_w + x + 1));
            x += 2;
            if (codes_index < num_codes) {
                codes.push_back(result);
                codes_index += 1;
            }
        } else if (i == 1) {
            result = ReLU(result);
        } else if (i == 2) {
            result = sigmoid(result);
        } else {
            std::cout << "Layer not imp yet";
            break;
        }
    }

    return {result, codes};
}

NB_MODULE(nanobind_get_codes, m) {
    m.def("get_codes", &get_codes);
    m.doc() = "Forward pass in c++";
}