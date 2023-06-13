#include <math.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
// #include <torch/torch.h>

#include <iostream>
#include <vector>

namespace nb = nanobind;

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
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> lin(
    const nb::DRef<Eigen::MatrixXd> &input,
    const nb::DRef<Eigen::MatrixXd> &weight,
    const nb::DRef<Eigen::VectorXd> &bias) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> res =
        (input * weight.transpose());
    for (size_t i = 0; i < res.rows(); ++i) { res.row(i) += bias; }
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

double BCELoss(const nb::DRef<Eigen::MatrixXd> &predictions,
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

double MSELoss(const nb::DRef<Eigen::MatrixXd> &predictions,
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
        sum *= (1.0 / static_cast<double>(M));
        tot += sum;
    }

    return tot / static_cast<double>(N);
}

void log_softmax(nb::DRef<Eigen::MatrixXd> input){
    int    N   = input.rows();
    int    M   = input.cols();
    double sum_exps = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        std::vector<double> exps;
        for (size_t j = 0; j < M; ++j) {
            exps.push_back(std::exp(input(i,j)));
        }
        sum_exps = 0.0;
        for (auto exp : exps)
            sum_exps += exp;

        for (size_t j = 0; j < M; ++j) {
            input(i,j) = std::log(exps[j] / sum_exps);
        }
    }

}

void softmax(nb::DRef<Eigen::MatrixXd> input){
    int    N   = input.rows();
    int    M   = input.cols();
    double sum_exps = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        std::vector<double> exps;
        for (size_t j = 0; j < M; ++j) {
            exps.push_back(std::exp(input(i,j)));
        }
        sum_exps = 0.0;
        for (auto exp : exps)
            sum_exps += exp;

        for (size_t j = 0; j < M; ++j) {
            input(i,j) =(exps[j] / sum_exps);
        }
    }

}

Eigen::MatrixXd one_hot_encoding(const nb::DRef<Eigen::VectorXi> &input, int num_classes){
    int N = input.rows();
    int M = num_classes;
    Eigen::MatrixXd res(N, M);
    for (int i = 0; i < N; i++){
        for (int j = 0; j<M ; j++){
            if (input[i] == j){
                res(i,j) = 1;
            }else{
                res(i,j) = 0;
            }
        }
    }
    return res;
}

double negative_log_likelihood(const nb::DRef<Eigen::MatrixXd> &log_likelihoods, const nb::DRef<Eigen::VectorXi> &targets){
    double sum = 0.0;
    for(int i =0 ; i < log_likelihoods.rows(); i++){
        sum+= log_likelihoods(i,targets(i));
    }
    return (-1.0/log_likelihoods.rows())*sum;
}

double cross_entropy_loss(nb::DRef<Eigen::MatrixXd> input, const nb::DRef<Eigen::VectorXi> &targets){
    log_softmax(input);
    return negative_log_likelihood(input, targets);
}

Eigen::MatrixXd differentiate_sigmoid(const nb::DRef<Eigen::MatrixXd> &x) {
    Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0);
    Eigen::MatrixXd tmp  = sigmoid(x);
    Eigen::MatrixXd res  = tmp.cwiseProduct(ones - tmp);
    return res;
}

Eigen::MatrixXd differentiate_ReLU(const nb::DRef<Eigen::MatrixXd> &x) {
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

Eigen::MatrixXd differentiate_BCELoss(const nb::DRef<Eigen::MatrixXd> &output,
                                      const nb::DRef<Eigen::MatrixXd> &target) {
    int             N    = output.rows();
    int             M    = output.cols();
    double          eps  = 1e-12;
    double          norm = -1.0 / ((double)M * (double)N);

    Eigen::MatrixXd tmp  = Eigen::MatrixXd::Constant(N, M, 1.0 + eps);
    Eigen::MatrixXd res =
        norm *
        ((target - output)
             .cwiseQuotient(
                 (tmp - output).cwiseProduct((output.array() + eps).matrix())));
    return res;
}

Eigen::MatrixXd differentiate_MSELoss(const nb::DRef<Eigen::MatrixXd> &output,
                                      const nb::DRef<Eigen::MatrixXd> &target) {
    int             N    = output.rows();
    int             M    = output.cols();
    double          norm = (2.0 / ((double)M * (double)N));

    Eigen::MatrixXd res  = norm * (output - target);
    return res;
}

Eigen::MatrixXd differentiate_CrossEntropyLoss(nb::DRef<Eigen::MatrixXd> output,
                                      const nb::DRef<Eigen::VectorXi> &target, int num_classes){
    softmax(output);
    return (output- one_hot_encoding(target, num_classes))/output.rows();
}

 
// template <typename Derived>
// void test_eigen_base(const nb::DRef<Eigen::EigenBase<Derived> &targets){
//     std::cout << targets << std::endl;
// }

// // Doesn't really need to be a function but for completness
// Breaks build
// std::vector<Eigen::MatrixXd> differentiate_linear_layer(
//     nb::DRef<Eigen::MatrixXd> &input) {
//     int                          N = input.rows();
//     int                          M = input.cols();

//     Eigen::MatrixXd dw = input;
//     std::vector<Eigen::MatrixXd> grads;
//     // dw
//     grads.push_back(dw);
//     // db - This is wront dim
//     grads.push_back(Eigen::MatrixXd::Constant(N, M, 1.0));
//     return grads;
// }
