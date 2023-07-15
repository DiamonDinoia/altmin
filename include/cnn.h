// //
// // Created by mbarbone on 6/26/23.
// //

// //
// // Edited by Tom 6/07/23
// // Completly rework neural network class

// #ifndef FAST_ALTMIN_CNN_H
// #define FAST_ALTMIN_CNN_H

// #include "cnn_layers.h"
// #include <nanobind/eigen/dense.h>
// #include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
// #include <nanobind/stl/bind_map.h>
// #include <nanobind/stl/string.h>
// #include <nanobind/stl/tuple.h>
// #include <nanobind/stl/shared_ptr.h>
// #include <nanobind/stl/unique_ptr.h>


// #include <cmath>
// #include <iostream>
// #include <utility>
// #include <vector>
// #include <tuple>
// #include <thread>


// enum loss_t { BCE, MSE, CrossEntropy };
// template <loss_t loss>
// class VariantCNN{
// public:
//     VariantCNN(){
//         const auto n_threads = std::thread::hardware_concurrency();
//         Eigen::setNbThreads(n_threads);
        
//     };
//     std::vector<std::variant<Conv2dLayer, ReluCNNLayer, MaxPool2dLayer>> layers;
//     std::vector<std::tuple<int,int>> weight_pairs; 
//     bool init_vals = true;

//     //Copy for now
//     ALTMIN_INLINE void addConv2dLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, Eigen::VectorXd bias, int batch_size, int C_in, int height, int width) noexcept{
//         layers.emplace_back(Conv2dLayer{kernels, bias, batch_size, C_in, height, width});
//     }

//     ALTMIN_INLINE void addMaxPool2dLayer(int kernel_size, int stride, int batch_size, int C, int height, int width) noexcept{
//         layers.emplace_back(MaxPool2dLayer{kernel_size, stride, batch_size, C, height, width});
//     }
    

//     // ALTMIN_INLINE void addLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
//     //     const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
//     //         layers.emplace_back(LinearLayer{batch_size, weight, bias, learning_rate});
//     // }

//     // ALTMIN_INLINE void addLastLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
//     //     const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
//     //         layers.emplace_back(LastLinearLayer{batch_size, weight, bias, learning_rate});
//     // }

//     ALTMIN_INLINE void addReluCNNLayer(const int N, const int C, const int H, const int W) noexcept {
//         layers.emplace_back(ReluCNNLayer{N, C, H, W});
//     }

//     // ALTMIN_INLINE void addSigmoidLayer(const int n, const int batch_size) noexcept {
//     //     layers.emplace_back(SigmoidLayer{n, batch_size});
//     // }

//     // ALTMIN_INLINE void construct_pairs() noexcept {
//     //     int prev_idx = 0;
//     //     for (int x = 0; x<layers.size(); x++){
//     //         if (layers[x].index() == 0 || layers[x].index() == 1 ){
//     //             weight_pairs.emplace_back(prev_idx,x);
//     //             prev_idx = x;
//     //         }
//     //     }
//     // }


//     ALTMIN_INLINE std::vector<std::vector<Eigen::MatrixXd>> get_codes_cnn(const std::vector<std::vector<Eigen::MatrixXd>> &inputs, const bool training_mode) noexcept {
//         std::visit(CallForwardCNN<decltype(inputs)>{inputs}, layers[0]);
//         for (int x = 1; x < layers.size();x++){
//             std::cout << x << std::endl;
//             std::vector<std::vector<Eigen::MatrixXd>> in = std::visit(get_layer_output_cnn, layers[x-1]);
//             std::visit(CallForwardCNN<decltype(in)>{in}, layers[x]);
//         }
//         //std::vector<std::vector<Eigen::MatrixXd>> res = std::visit(get_layer_output_cnn, layers[0]);
//         return std::visit(get_layer_output_cnn, layers[layers.size()-1]);
      
//     }


// };

// #endif  // FAST_ALTMIN_NEURAL_NETWORK_H
