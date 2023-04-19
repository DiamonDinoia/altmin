#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

// I'm passing a reference to the data location not the data itself
// So I think it makes the most sense to transpose it here as
// nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
// Eigen::RowMajor>> &weight doesn't change the order of the data as a
// referenced is passed to the data And it's quicker to transpose the data here
// than in cpp But i'll come nack to this at the end
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lin(
    const nb::DRef<Eigen::MatrixXd> &input,
    const nb::DRef<Eigen::MatrixXd> &weight,
    const nb::DRef<Eigen::MatrixXd> &bias) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res =
        (input * weight.transpose());
    for (size_t i = 0; i < res.rows(); ++i) { res.row(i) += bias.transpose(); }
    return res;
}

// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
void ReLU(nb::DRef<Eigen::MatrixXd> input) {
    int N = input.rows();
    int M = input.cols();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            input(i, j) = std::max(input(i, j), 0.0);
        }
    }
    return;
}

// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
void sigmoid(nb::DRef<Eigen::MatrixXd> input) {
    int N = input.rows();
    int M = input.cols();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            input(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return;
}