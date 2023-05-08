#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <vector>

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

Eigen::MatrixXd ReLU(nb::DRef<Eigen::MatrixXd> input) {
    int             N = input.rows();
    int             M = input.cols();
    Eigen::MatrixXd res(N, M);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            res(i, j) = std::max(input(i, j), 0.0);
        }
    }
    return res;
}

// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
void ReLU_inplace(nb::DRef<Eigen::MatrixXd> input) {
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
Eigen::MatrixXd sigmoid(nb::DRef<Eigen::MatrixXd> input) {
    int             N = input.rows();
    int             M = input.cols();
    Eigen::MatrixXd res(N, M);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            res(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return res;
}

// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
void sigmoid_inplace(nb::DRef<Eigen::MatrixXd> input) {
    int N = input.rows();
    int M = input.cols();
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            input(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return;
}

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

float MSELoss(const nb::DRef<Eigen::MatrixXd> &predictions,
              const nb::DRef<Eigen::MatrixXd> &targets) {
    int    N   = predictions.rows();
    int    M   = predictions.cols();
    double tot = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < M; ++j) {
            sum += (targets(i, j) - predictions(i, j)) *
                   (targets(i, j) - predictions(i, j));
        }
        sum *= (1.0 / static_cast<float>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}
