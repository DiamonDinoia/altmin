// #ifndef FAST_ALTMIN_CNN_LAYERS_H
// #define FAST_ALTMIN_CNN_LAYERS_H

// #include "defines.h"
// #include "functions.hpp"
// #include "types.h"
// #include <thread>
// #include <variant>

// //Need to init layer outputs properly 
// //Worth having a seperate conv network class
// class Conv2dLayer{
// public:
//     std::vector<std::vector<Eigen::MatrixXd>> kernels; 
//     std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
//     Eigen::VectorXd bias; 
//     int N;
//     int C_in;
//     int C_out;
//     int H;
//     int W;

//     Conv2dLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, Eigen::VectorXd bias, int N, int C_in, int H, int W) : kernels(kernels), bias(bias), N(N), C_in(C_in), C_out(kernels.size()), H(H), W(W){
//         for (int n = 0; n < N; n++){
//             std::vector<Eigen::MatrixXd> tmp;
//             for (int c = 0; c < C_out; c++){
//                 tmp.emplace_back(Eigen::MatrixXd(H,W));
//             }
//             layer_outputs.emplace_back(tmp);
//         } 
//     }

//     //works for one channel in atm
//     template <typename T>
//     ALTMIN_INLINE void forward(const T& inputs) noexcept{
//         for (int n = 0; n < N; n++){
//             for (int c_out = 0; c_out < C_out; c_out++){
//                 Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(H,W);
//                 //Bias only needs to be added once so do like this so its done during the first convolution so we don't have to iterate over the whole matrix again at the end.
//                 double bias_val = bias[c_out];
//                 for (int c_in = 0; c_in < C_in; c_in++){
//                     sum +=  conv2d(inputs[n][c_in], kernels[c_out][c_in], bias_val, H, W);
//                     bias_val = 0.0;
//                 }
                
//                 layer_outputs[n][c_out] = sum;
//             }
//         }      
//     }
// };

// class ReluCNNLayer{
// public:

//     std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
//     int N;
//     int C;
//     int H;
//     int W;
//     ReluCNNLayer(const int N, const int C, const int H, const int W) : N(N), C(C), H(H), W(W){
//         for (int n = 0; n < N; n++){
//             std::vector<Eigen::MatrixXd> tmp;
//             for (int c = 0; c < C; c++){
//                 tmp.emplace_back(Eigen::MatrixXd(H,W));
//             }
//             layer_outputs.emplace_back(tmp);
//         } 
//     };
//     template <typename T>
//     ALTMIN_INLINE void forward(const T& inputs) noexcept{
//         for (int n = 0; n < N; n++){
//             for (int c = 0; c < C; c++){
//                 layer_outputs[n][c] = ReLU(inputs[n][c]);
//             }
//         }   
//     }
// };

// class MaxPool2dLayer{
// public: 
//     std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
//     int kernel_size;
//     int stride;
//     int N;
//     int C;
//     int H;
//     int W;

//     MaxPool2dLayer(int kernel_size, int stride, int N, int C, int H, int W) :  kernel_size(kernel_size), stride(stride),  N(N), C(C), H(H), W(W){
//         for (int n = 0; n < N; n++){
//             std::vector<Eigen::MatrixXd> tmp;
//             for (int c = 0; c < C; c++){
//                 tmp.emplace_back(Eigen::MatrixXd(H, W));
//             }
//             layer_outputs.emplace_back(tmp);
//         } 
//     }

//     //works for one channel in atm
//     template <typename T>
//     ALTMIN_INLINE void forward(const T& inputs) noexcept{
//         for (int n = 0; n < N; n++){
//             for (int c = 0; c < C; c++){
//                 layer_outputs[n][c] = maxpool2d(inputs[n][c], kernel_size, stride, H, W);
//             }
//         }      
//     }
// };

// template <typename T>
// struct CallForwardCNN {
//     void operator()(Conv2dLayer& conv2d) { conv2d.forward(inputs); }    
//     void operator()(ReluCNNLayer& relu_cnn) { relu_cnn.forward(inputs); }   
//     void operator()(MaxPool2dLayer& maxpool2d) { maxpool2d.forward(inputs); }   
//     const T& inputs;
// };

// //std::vector<std::vector<Eigen::MatrixXd>>
// const auto&  get_layer_output_cnn = [](const auto& layer) {  return layer.layer_outputs;};



// #endif  // FAST_ALTMIN_NEURAL_NETWORK_H