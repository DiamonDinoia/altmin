#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <vector>

#include "nanobind_layers.hpp"

namespace nb = nanobind;

/*
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
differentiate_last_layers(const nb::DRef<Eigen::MatrixXd> &c) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res =
        // sigmoid(W^3 Relu(c)) (1- sigmoid(W^3Relu(c))) w^3 Relu^'(c)
        tmp = sigmoid(lin(ReLU(c), weights, biases));
         return res;
}
*/

Eigen::MatrixXd differentiate_sigmoid(nb::DRef<Eigen::MatrixXd> x) {
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0);
    Eigen::MatrixXd tmp  = sigmoid(x);
    return tmp * (ones - tmp);
}

Eigen::MatrixXd differentiate_ReLU(nb::DRef<Eigen::MatrixXd> x) {
    int             N = x.rows();
    int             M = x.cols();
    Eigen::MatrixXd res(N, M);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            if (x(i, j) >= 0.0) {
                res(i, j) = 1.0;
            } else {
                res(i, j) = 0.0;
            }
        }
    }
    return res;
}

Eigen::MatrixXd differentiate_last_layer(
    const nb::DRef<Eigen::MatrixXd> &x,
    const nb::DRef<Eigen::MatrixXd> &weights) {
    Eigen::MatrixXd tmp = weights * ReLU(x);
    Eigen::MatrixXd one = differentiate_sigmoid(tmp);
    Eigen::MatrixXd two = weights * differentiate_ReLU(x);
    return one * two;
    // return x;
}

float differentiate_mse(const nb::DRef<Eigen::MatrixXd> &predictions,
                        const nb::DRef<Eigen::MatrixXd> &targets) {
    int             N   = predictions.rows();
    int             M   = predictions.cols();
    double          tot = 0.0;
    Eigen::MatrixXd weight_derivative(N, M);

    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < M; ++j) {
            weight_derivative(i, j) = (targets(i, j) - predictions(i, j));
        }
        sum *= (1.0 / static_cast<float>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}
