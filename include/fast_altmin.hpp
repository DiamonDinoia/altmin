#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <iostream>
#include <vector>

#define ALTMIN_INLINE inline __attribute__((always_inline))

////////////////////////////////////////////////////////////////////////////////////////


// No functions from in here actually used atm as this is mostly eigen stuff
// ////////////

/////////////////////////////////////////////////////////////////////////////////////////

// I'm passing a reference to the data location not the data itself
// So I think it makes the most sense to transpose it here as
// nb::DRef<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
// Eigen::RowMajor>> &weight doesn't change the order of the data as a
// referenced is passed to the data And it's quicker to transpose the data here
// than in cpp But i'll come nack to this at the end
ALTMIN_INLINE Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lin(
        const nanobind::DRef<Eigen::MatrixXd> &input,
        const nanobind::DRef<Eigen::MatrixXd> &weight,
        const nanobind::DRef<Eigen::VectorXd> &bias) noexcept {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res =
            (input * weight.transpose());
    for (long i = 0; i < res.rows(); ++i) { res.row(i) += bias; }
    return res;
}


// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
ALTMIN_INLINE void ReLU_inplace(nanobind::DRef<Eigen::MatrixXd> input) noexcept {
    const auto N = input.rows();
    const auto M = input.cols();
    for (auto i = 0L; i < N; ++i) {
        for (auto j = 0L; j < M; ++j) {
            input(i, j) = std::max(input(i, j), 0.0);
        }
    }
}

ALTMIN_INLINE Eigen::MatrixXd ReLU(const nanobind::DRef<Eigen::MatrixXd> &input) noexcept {
    Eigen::MatrixXd res = input;
    ReLU_inplace(res);
    return res;
}



// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
ALTMIN_INLINE Eigen::MatrixXd sigmoid(const nanobind::DRef<Eigen::MatrixXd> &input) noexcept {
    const auto N = input.rows();
    const auto M = input.cols();
    Eigen::MatrixXd res(N, M);
    for (auto i = 0l; i < N; ++i) {
        for (auto j = 0l; j < M; ++j) {
            res(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
    return res;
}

// Maybe should iterate in the opposite direction as should go in storage order
// but again I'll come back to this
// Also note the reference to the data is passed and the data is changed in
// place so nothing returned
ALTMIN_INLINE void sigmoid_inplace(nanobind::DRef<Eigen::MatrixXd> input) noexcept {
    const auto N = input.rows();
    const auto M = input.cols();
    for (auto i = 0l; i < N; ++i) {
        for (auto j = 0l; j < M; ++j) {
            input(i, j) = 1.0 / (1.0 + std::exp(-input(i, j)));
        }
    }
}

ALTMIN_INLINE double BCELoss(const nanobind::DRef<Eigen::MatrixXd> &predictions,
                             const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {
    const auto N = predictions.rows();
    const auto M = predictions.cols();
    double tot = 0.0;
    for (long i = 0; i < N; ++i) {
        double sum = 0.0;
        for (long j = 0; j < M; ++j) {
            sum += (targets.row(i)(j) * std::log(predictions.row(i)(j))) +
                   ((1.0 - targets.row(i)(j)) *
                    std::log(1.0 - predictions.row(i)(j)));
        }
        sum *= (-1.0 / static_cast<double>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}

ALTMIN_INLINE double MSELoss(const nanobind::DRef<Eigen::MatrixXd> &predictions,
                             const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {
    const auto N = predictions.rows();
    const auto M = predictions.cols();
    double tot = 0.0;
    for (long i = 0; i < N; ++i) {
        double sum = 0.0;
        for (long j = 0; j < M; ++j) {
            sum += (targets(i, j) - predictions(i, j)) *
                   (targets(i, j) - predictions(i, j));
        }
        sum *= (1.0 / static_cast<double>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}

ALTMIN_INLINE void log_softmax(nanobind::DRef<Eigen::MatrixXd> input) noexcept {
    const auto N = input.rows();
    const auto M = input.cols();
    // TODO: Probably can be done using eignen instead of std::vector and various loops
    std::vector<double> exps(M);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < M; ++j) {
            exps[j] = std::exp(input(i, j));
        }
        double sum_exps = 0.0;
        for (const auto exp: exps) {
            sum_exps += exp;
        }
        const auto inv_sum_exps = 1.0 / sum_exps;
        for (long j = 0; j < M; ++j) {
            input(i, j) = std::log(exps[j] * inv_sum_exps);
        }
    }

}

ALTMIN_INLINE void softmax(nanobind::DRef<Eigen::MatrixXd> &input) noexcept {
    const auto N = input.rows();
    const auto M = input.cols();
    std::vector<double> exps(M);

    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < M; ++j) {
            exps[j] = std::exp(input(i, j));
        }
        double sum_exps = 0.0;
        for (auto exp: exps) {
            sum_exps += exp;
        }
        const auto inv_sum_exps = 1.0 / sum_exps;
        for (long j = 0; j < M; ++j) {
            input(i, j) = (exps[j] * inv_sum_exps);
        }
    }

}

ALTMIN_INLINE Eigen::MatrixXd
one_hot_encoding(const nanobind::DRef<Eigen::VectorXd> &input, const int num_classes) noexcept {
    const auto N = input.rows();
    const auto M = num_classes;
    Eigen::MatrixXd res(N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            res(i, j) = static_cast<int>(input[i]) == j;
        }
    }
    return res;
}

ALTMIN_INLINE double negative_log_likelihood(const nanobind::DRef<Eigen::MatrixXd> &log_likelihoods,
                                             const nanobind::DRef<Eigen::VectorXi> &targets) noexcept {
    double sum = 0.0;
    for (int i = 0; i < log_likelihoods.rows(); i++) {
        sum += log_likelihoods(i, targets(i));
    }
    return (-1.0 / static_cast<double>(log_likelihoods.rows())) * sum;
}

ALTMIN_INLINE
double
cross_entropy_loss(nanobind::DRef<Eigen::MatrixXd> input, const nanobind::DRef<Eigen::VectorXi> &targets) noexcept {
    log_softmax(input);
    return negative_log_likelihood(input, targets);
}

ALTMIN_INLINE Eigen::MatrixXd differentiate_sigmoid(const nanobind::DRef<Eigen::MatrixXd> &x) noexcept {
    const auto ones = Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0);
    const auto tmp = sigmoid(x);
    return tmp.cwiseProduct(ones - tmp);
}

ALTMIN_INLINE Eigen::MatrixXd differentiate_ReLU(const nanobind::DRef<Eigen::MatrixXd> &x) noexcept {
    long N = x.rows();
    long M = x.cols();
    Eigen::MatrixXd res(N, M);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < M; ++j) {
            res(i, j) = x(i, j) >= 0.0;
        }
    }
    return res;
}

ALTMIN_INLINE Eigen::MatrixXd differentiate_BCELoss(const nanobind::DRef<Eigen::MatrixXd> &output,
                                                    const nanobind::DRef<Eigen::MatrixXd> &target) noexcept {
    const auto N = output.rows();
    const auto M = output.cols();
    const auto eps = 1e-12;
    const auto norm = -1.0 / ((double) M * (double) N);

    Eigen::MatrixXd tmp = Eigen::MatrixXd::Constant(N, M, 1.0 + eps);
    return norm * ((target - output).cwiseQuotient((tmp - output).cwiseProduct((output.array() + eps).matrix())));
}

ALTMIN_INLINE Eigen::MatrixXd differentiate_MSELoss(const nanobind::DRef<Eigen::MatrixXd> &output,
                                                    const nanobind::DRef<Eigen::MatrixXd> &target) noexcept {

    const auto N = output.rows();
    const auto M = output.cols();
    const auto norm = (2.0 / static_cast<double>(M * N));
    return norm * (output - target);
}

ALTMIN_INLINE Eigen::MatrixXd differentiate_CrossEntropyLoss(nanobind::DRef<Eigen::MatrixXd> output,
                                                             const nanobind::DRef<Eigen::VectorXd> &target,
                                                             const int num_classes) noexcept {
    softmax(output);
    return (output - one_hot_encoding(target, num_classes)) / output.rows();
}

