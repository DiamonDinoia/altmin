#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "defines.h"
#include "types.h"
#include <Eigen/Dense>
#include <thread>

template <typename T>
Eigen::MatrixXd flatten(T& inputs){
    int dim_0 = inputs.size();
    int dim_1 = inputs[0].size() * inputs[0][0].rows() * inputs[0][0].cols();
    int matrix_size = inputs[0][0].rows() * inputs[0][0].cols();
    Eigen::MatrixXd res(dim_0, dim_1);
    res.setZero(dim_0,dim_1);
    int row_pos = 0;
    for (int x = 0 ; x < inputs.size(); x++){
        int col_pos = 0;
        for (int y = 0; y < inputs[0].size(); y++){
            //defeats the point of templating
            //use decltype to handel case of T being list of vectors
            Eigen::MatrixXd tmp = inputs[x][y];

            res.block(row_pos,col_pos, 1, matrix_size) = tmp.reshaped<Eigen::RowMajor>().transpose();
            col_pos += matrix_size;
        }   
        row_pos+=1;
    }

    return res;
}

template <typename T>
Eigen::MatrixXd conv2d(const T& input, const T& kernel, double bias, const int height, const int width){
    Eigen::MatrixXd res(width, height);
    int kernel_size_rows = kernel.rows();
    int kernel_size_cols = kernel.cols();
    for (int i = 0; i < height; i++){
        for(int j=0; j < width; j++){
            res(i,j) = (input.block(i,j,kernel_size_rows,kernel_size_cols).cwiseProduct(kernel)).sum()+bias;
        }
    }
    return res;

}

template <typename T>
Eigen::MatrixXd maxpool2d(const T &input, const int kernel_size, const int stride, const int height, const int width){
    Eigen::MatrixXd res(width, height);
    int count_i = 0;
    for (int i = 0; i < height; i++){
        int count_j = 0; 
        for(int j=0; j < width; j++){
            res(i,j) = input.block(count_i,count_j,kernel_size,kernel_size).maxCoeff();
            count_j += stride;
        }
        count_i += stride;
    }
    return res;
}

//Chnages to this and bce loss have made it worse
ALTMIN_INLINE auto lin(const auto & input,
                       const auto & weight,
                       const auto & bias) noexcept {
    const auto n_threads = std::thread::hardware_concurrency();
    Eigen::setNbThreads(n_threads);
    Eigen::MatrixXd res = input * weight.transpose();
    for (long i = 0; i < res.rows(); ++i) { res.row(i) += bias; }
    return res;
}

template <typename T>
ALTMIN_INLINE auto ReLU(const T& input) noexcept {
    Eigen::MatrixXd res = input;
    for (auto i = 0L; i<input.size(); ++i){
        *(res.data()+i) =  std::max(*(res.data()+i), 0.0);
    }
    return res;
}

template <typename T>
ALTMIN_INLINE auto sigmoid(const T& input) noexcept {
    Eigen::MatrixXd res = input;
    for (auto i = 0L; i<input.size(); ++i){
        *(res.data()+i) =  1.0 / (1.0 + std::exp(-*(res.data()+i)));
    }
    return res;
}

template <typename T>
ALTMIN_INLINE double BCELoss(const T& predictions,
                             const T& targets) noexcept {
    const auto N = predictions.rows();
    const auto M = predictions.cols();
    double tot   = 0.0;
    for (long j = 0; j < M; ++j) {
        double sum = 0.0;
        for (long i = 0; i < N; ++i) {
            sum += (targets(i,j) * std::log(predictions(i,j))) +
                   ((1.0 - targets(i,j)) * std::log(1.0 - predictions(i,j)));
        }
        sum *= (-1.0 / static_cast<double>(N));
        tot += sum;
    }
    return tot / static_cast<double>(M);
}

// template <typename T>
// ALTMIN_INLINE double BCELoss(const T& predictions,
//                              const T& targets) noexcept {
//     const auto N = predictions.rows();
//     const auto M = predictions.cols();
//     double tot   = 0.0;
//     for (long i = 0; i < N; ++i) {
//         double sum = 0.0;
//         for (long j = 0; j < M; ++j) {
//             sum += (targets(i,j) * std::log(predictions(i,j))) +
//                    ((1.0 - targets(i,j)) * std::log(1.0 - predictions(i,j)));
//         }
//         sum *= (-1.0 / static_cast<double>(M));
//         tot += sum;
//     }
//     return tot / static_cast<double>(N);
// }

template <typename T>
ALTMIN_INLINE double MSELoss(const T& predictions,
                             const T& targets) noexcept {
    const auto N = predictions.rows();
    const auto M = predictions.cols();
    double tot   = 0.0;
    for (long j = 0; j < M; ++j) {
        double sum = 0.0;
        for (long i = 0; i < N; ++i) {
            sum += (targets(i, j) - predictions(i, j)) * (targets(i, j) - predictions(i, j));
        }
        sum *= (1.0 / static_cast<double>(M));
        tot += sum;
    }
    return tot / static_cast<double>(N);
}

template <typename T>
ALTMIN_INLINE void log_softmax(T& input) noexcept {
    for (auto row : input.rowwise()){
        row = (row.array().exp() * (1/row.array().exp().sum())).log();
    }
}

template <typename T>
ALTMIN_INLINE void softmax(T& input) noexcept {
    for (auto row : input.rowwise()){
        row = row.array().exp() * (1/row.array().exp().sum());
    }
}

template <typename T>
ALTMIN_INLINE auto one_hot_encoding(const T& input, const int num_classes) noexcept {
    const auto N = input.rows();
    const auto M = num_classes;
    Eigen::MatrixXd res(N, M);
    for (long j = 0; j < M; ++j) {
        for (long i = 0; i < N; ++i) {
            res(i, j) = static_cast<int>(input[i]) == j; 
        }

    }
    return res;
}

template <typename T, typename G>
ALTMIN_INLINE double negative_log_likelihood(const T& log_likelihoods,
                                             const G& targets) noexcept {
    double sum = 0.0;
    for (int i = 0; i < log_likelihoods.rows(); i++) {
        sum += log_likelihoods(i, targets(i));
    }
    return (-1.0 / static_cast<double>(log_likelihoods.rows())) * sum;
}

template <typename T, typename G>
ALTMIN_INLINE double cross_entropy_loss(const T& input,
                                        const G& targets) noexcept {
    auto res = input;
    log_softmax(res);
    return negative_log_likelihood(res, targets);
}


template <typename T>
ALTMIN_INLINE auto differentiate_sigmoid(T &x) noexcept {
    const auto ones = Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0);
    auto res = sigmoid(x);
    return res.cwiseProduct(ones - res);
}

template <typename T>
ALTMIN_INLINE auto differentiate_ReLU(const T &x) noexcept {
    long N = x.rows();
    long M = x.cols();
    Eigen::MatrixXd res(N, M);
    for (long j = 0; j < M; ++j) {
        for (long i = 0; i < N; ++i) {
            res(i, j) = x(i, j) >= 0.0;
        }
    }
    return res;
}

template <typename T>
Eigen::MatrixXd differentiate_maxpool2d(const T &input, const int kernel_size, const int stride, const int height, const int width){
    Eigen::MatrixXd res(input.rows(), input.cols());
    res.setZero(input.rows(), input.cols());
   
    int count_i = 0;
    for (int i = 0; i < height; i++){
        int count_j = 0; 
        for(int j=0; j < width; j++){
            Eigen::Index maxRow, maxCol;

            // pytorch argmax and maxCoeff both give index of first element if there are multiple elements with the same max value
            // but argmax goes in row major order and maxcoeff goes in col major order so this can result in giving different indices as it leads to a different element being the first element seen for max value. 
            // Either way is fine for computing the gradient but they give different gradients which means I can't unit test the implementation without doing the workaround below
            // Current plan is to template with bool so that only does expensive workaround for unit tests
            Eigen::MatrixXd tmp = input.block(count_i,count_j,kernel_size,kernel_size);
            Eigen::VectorXd maxVec = tmp.rowwise().maxCoeff();
            maxVec.maxCoeff(&maxRow);
            Eigen::Index dummy;
            tmp.row(maxRow).maxCoeff(&dummy, &maxCol);
            res(count_i + maxRow, count_j + maxCol) = 1.0;

            count_j += stride;
        }
        count_i += stride;
    }
     
    return res;
}



//template with bool to indicate what partial derivative to calc
template <typename T>
Eigen::MatrixXd differentiate_conv2d(const T& input, const T& kernel, const int height, const int width, bool flag){
    //dL_df
    //kernel is dL_dO output from next layer
    if (flag){
        Eigen::MatrixXd res(width, height);
        int kernel_size_rows = kernel.rows();
        int kernel_size_cols = kernel.cols();
        for (int i = 0; i < height; i++){
            for(int j=0; j < width; j++){
                res(i,j) = (input.block(i,j,kernel_size_rows,kernel_size_cols).cwiseProduct(kernel)).sum();
            }
        }
        return res;
    }else{
        int kernel_size_rows = kernel.rows();
        int kernel_size_cols = kernel.cols();
        Eigen::MatrixXd input_padded (input.rows()+(kernel_size_rows*2)-2, input.cols()+(kernel_size_cols*2)-2);
        input_padded.setZero();
        input_padded.block(kernel_size_rows-1,kernel_size_cols-1, input.rows(), input.cols()) = input; 
       

        Eigen::MatrixXd res(width, height);
       
        Eigen::MatrixXd rotated_filter = kernel.transpose().colwise().reverse().transpose().colwise().reverse();

        for (int i = 0; i < height; i++){
            for(int j=0; j < width; j++){
                res(i,j) = (input_padded.block(i,j,kernel_size_rows,kernel_size_cols).cwiseProduct(rotated_filter)).sum();
            }
        }
        //std::cout << "Res " << res << std::endl;
        return res;

    }
    

}

//both matirxXd but targets nanobind ref and output normal ref
template <typename T, typename G>
ALTMIN_INLINE auto differentiate_BCELoss(const T& output,
                                                    const G& target) noexcept {
    const auto N = output.rows();
    const auto M = output.cols();
    const auto eps = 1e-12;
    const auto norm = -1.0 / ((double) M * (double) N);
    const auto tmp = Eigen::MatrixXd::Constant(N, M, 1.0 + eps);
    return norm * ((target - output).cwiseQuotient((tmp - output).cwiseProduct((output.array() + eps).matrix())));
}

template <typename T, typename G>
ALTMIN_INLINE auto differentiate_MSELoss(const T& output,
                                                    const G& target) noexcept {
    const auto N = output.rows();
    const auto M = output.cols();
    const auto norm = (2.0 / static_cast<double>(M * N));
    return norm * (output - target);
}

template <typename T, typename G>
    ALTMIN_INLINE auto differentiate_MSELoss4d(const T& output,
                                                    const G& target) noexcept{
        const int N = output.size();
        const int C = output[0].size();
        std::vector<std::vector<Eigen::MatrixXd>> dc; 
        for (int n = 0; n < N; n++){
            std::vector<Eigen::MatrixXd> tmp; 
            for (int c = 0; c < C; c++){
                tmp.emplace_back(differentiate_MSELoss(output[n][c], target[n][c])/static_cast<double>(N*C));
            }
            dc.emplace_back(tmp);
        }   
        return dc;
    }

template <typename T, typename G>
ALTMIN_INLINE auto differentiate_CrossEntropyLoss(const T& output,
                                                             const G& target,
                                                             const int num_classes) noexcept {
    Eigen::MatrixXd res = output;
    softmax(res);
    Eigen::MatrixXd tmp = one_hot_encoding(target, num_classes);
    Eigen::MatrixXd res_two = res - tmp; 
    return (res_two) / res.rows();
}

