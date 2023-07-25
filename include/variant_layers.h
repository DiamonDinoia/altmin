//todo add const where possible 
//for adam use ref instead or auto 
//Try optimise elsewhere
//Make matrices class member if possible


#ifndef FAST_ALTMIN_LAYERS_H
#define FAST_ALTMIN_LAYERS_H

#include "defines.h"
#include "functions.hpp"
#include "types.h"
#include <thread>
#include <variant>

class Adam {
    public:

    Adam(   const int n,
            const int m,
            const double learning_rate,
            float beta1 = 0.9,
            float beta2 = 0.999,
            float eps   = 1e-08)
        : learning_rate(learning_rate),
            beta_1(beta1),
            beta_2(beta2),
            eps(eps),
            weight_m(Eigen::MatrixXd::Zero(n,m)),
            weight_v(Eigen::MatrixXd::Zero(n,m)),
            bias_m(Eigen::VectorXd::Zero(n)),
            bias_v(Eigen::VectorXd::Zero(n)),
            weight_m_t_correct(Eigen::MatrixXd::Zero(n,m)),
            weight_v_t_correct(Eigen::MatrixXd::Zero(n,m)),
            bias_m_t_correct(Eigen::VectorXd::Zero(n)),
            bias_v_t_correct(Eigen::VectorXd::Zero(n)){const auto n_threads = std::thread::hardware_concurrency();
            Eigen::setNbThreads(n_threads);}

    template <bool init_vals, bool update_bias, typename T, typename G>
    ALTMIN_INLINE void adam(Eigen::MatrixXd& weight, Eigen::VectorXd& bias, const T& grad_weight,
                            const G& grad_bias) noexcept {
       
        // Update weight
        if constexpr(init_vals) {
            weight_m = (1 - beta_1) * grad_weight;
            weight_v = (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        } else {
            weight_m = beta_1 * weight_m + (1 - beta_1) * grad_weight;
            weight_v = beta_2 * weight_v + (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        }

        
        weight_m_t_correct = weight_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        weight_v_t_correct = weight_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));

        
        weight -= learning_rate * (weight_m_t_correct.cwiseQuotient((weight_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        step += 1;

        // Update bias
        if constexpr(update_bias){
            if constexpr(init_vals){
                bias_m = (1 - beta_1) * grad_bias;
                bias_v = (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
            } else {
                bias_m = beta_1 * bias_m + (1 - beta_1) * grad_bias;
                bias_v = beta_2 * bias_v + (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
            }

    
            bias_m_t_correct = bias_m / (1.0 - std::pow(beta_1, static_cast<double>(bias_step)));
            bias_v_t_correct = bias_v / (1.0 - std::pow(beta_2, static_cast<double>(bias_step)));
    
            

            bias -= learning_rate * (bias_m_t_correct.cwiseQuotient((bias_v_t_correct.cwiseSqrt().array() + eps).matrix()));
            bias_step += 1;
        }
        
    }
   private:
    // See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html for
    // implementation details


    
    Eigen::MatrixXd weight_m;
    Eigen::MatrixXd weight_v;
    Eigen::VectorXd bias_m;
    Eigen::VectorXd bias_v;
    Eigen::MatrixXd weight_m_t_correct;
    Eigen::MatrixXd weight_v_t_correct;
    Eigen::VectorXd bias_m_t_correct;
    Eigen::VectorXd bias_v_t_correct;
    double learning_rate;
    float beta_1;
    float beta_2;
    float eps;
    int step = 1;
    int bias_step = 1;
};

enum layer_type {
        RELU, LINEAR, SIGMOID, LAST_LINEAR
    };

class Layer{
public:
    Eigen::MatrixXd layer_output;
    layer_type layer;

    int n;
    int batch_size;
    Layer(layer_type layer, const int n, const int batch_size) : layer(layer), n(n), batch_size(batch_size), layer_output(batch_size, n) {const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);}
    
   


};



class LinearLayer : public Layer{
public:

    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    Eigen::MatrixXd codes;
    double weights_learning_rate;
    double codes_learning_rate;
    Adam adam_optimiser;
    bool init_vals = true;

    LinearLayer(const int batchSize, Eigen::MatrixXd weight, Eigen::VectorXd bias, const double weights_learning_rate, const double codes_learning_rate):
        Layer(layer_type::LINEAR, weight.rows(), batchSize), weight(weight), bias(bias), codes(batch_size,weight.cols()),  weights_learning_rate(weights_learning_rate), codes_learning_rate(codes_learning_rate), 
        adam_optimiser(weight.rows(), weight.cols(), weights_learning_rate) {}

    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const bool store_codes) noexcept{
        layer_output = lin(inputs, weight, bias);
        if (store_codes){
            codes = layer_output;
        }
        
    }
   
    template <typename T>
    ALTMIN_INLINE void update_codes(const T &dc) noexcept{
        codes -= (((1.0 + 0.9) * codes_learning_rate) * dc);
    }

    ALTMIN_INLINE void set_codes(const nanobind::DRef<Eigen::MatrixXd> &codes) noexcept{
        this->codes = codes;
    };


    template <typename T, typename G>
    ALTMIN_INLINE void adam(const T& grad_weight, const G& grad_bias) noexcept {
        if (init_vals){
            adam_optimiser.template adam<true, true>(weight, bias, grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false, true>(weight, bias, grad_weight, grad_bias);
        }
        init_vals=false;
    }
   



    
};

class LastLinearLayer : public Layer{
public:

    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    double learning_rate;
    Adam adam_optimiser;
    bool init_vals=true;

    LastLinearLayer(const int batchSize, Eigen::MatrixXd weight, Eigen::VectorXd bias, const double learning_rate) :
        Layer(layer_type::LAST_LINEAR, weight.rows(), batchSize), weight(weight), bias(bias), learning_rate(learning_rate), 
        adam_optimiser(weight.rows(), weight.cols(), learning_rate) {}

    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const bool store_codes = false) noexcept{
        layer_output = lin(inputs, weight, bias);
    }


    template <typename T, typename G>
    ALTMIN_INLINE void adam(const T& grad_weight, const G& grad_bias) noexcept {

        if (init_vals){
            adam_optimiser.template adam<true, true>(weight, bias, grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false,true >(weight, bias, grad_weight, grad_bias);
        }
        
        init_vals = false;
    }

};

class ReluLayer : public Layer{
public:
    Eigen::MatrixXd dout;
    ReluLayer(const int n, const int batch_size): Layer(layer_type::RELU, n,batch_size), dout(batch_size, n){};
    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const  bool store_codes = false) noexcept{
        layer_output = ReLU(inputs);
    }
    template <typename T>
    ALTMIN_INLINE void differentiate_layer(const T& inputs) noexcept{
        dout = differentiate_ReLU(inputs);
    }

        
};

class SigmoidLayer : public Layer{
public:
    Eigen::MatrixXd dout;
    SigmoidLayer(const int n, const int batch_size): Layer(layer_type::SIGMOID, n,batch_size), dout(batch_size, n){};
    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const bool store_codes = false) noexcept{
        layer_output = sigmoid(inputs);
    }
    template <typename T>
    ALTMIN_INLINE void differentiate_layer(const T& inputs) noexcept{
        dout = differentiate_sigmoid(inputs);
    }
};

//test which is quicker
//autoget_layer_output = [](const auto& layer) {  return layer.layer_output;};
const auto& get_layer_output = [](const auto& layer) {  return layer.layer_output;};


//Fix store codes later;
template <typename T>
struct CallForward {
    void operator()(LinearLayer& lin) { lin.forward(inputs, train); }    
    void operator()(LastLinearLayer& last_lin) { last_lin.forward(inputs, false); }    
    void operator()(ReluLayer& relu) { relu.forward(inputs, false);}
    void operator()(SigmoidLayer& sigmoid) {sigmoid.forward(inputs, false);}
    const T& inputs;
    const bool train = true;
};

template <typename T>
struct CallDifferentiateLayer {
    void operator()(LinearLayer& lin) { std::cout << "differentiate layer called on linear layer\n"; }    
    void operator()(LastLinearLayer& last_lin) { std::cout << "differentiate layer called on last linear layer\n"; }    
    void operator()(ReluLayer& relu) { relu.differentiate_layer(inputs);}
    void operator()(SigmoidLayer& sigmoid) {sigmoid.differentiate_layer(inputs);}
    const T& inputs;
};

template <typename T>
struct CallUpdateCodes {
    void operator()(LinearLayer& lin) { lin.update_codes(dc); }    
    void operator()(LastLinearLayer& last_lin) { std::cout << "update codes called on last linear layer\n"; }    
    void operator()(ReluLayer& relu) {  std::cout << "update codes called on relu layer\n";}
    void operator()(SigmoidLayer& sigmoid) { std::cout << "update codes called on sigmoid layer\n";}
    const T& dc;
};

template <typename T, typename G>
struct CallAdam {
    void operator()(LinearLayer& lin) { lin.adam(dw,db); }    
    void operator()(LastLinearLayer& last_lin) { last_lin.adam(dw,db); }    
    void operator()(ReluLayer& relu) {  std::cout << "adam called on relu layer\n";}
    void operator()(SigmoidLayer& sigmoid) { std::cout << "adam called on sigmoid layer\n";}
    const T& dw;
    const G& db;
};

struct CallGetCodes {
    Eigen::MatrixXd& operator()(LinearLayer& lin) { return lin.codes; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    Eigen::MatrixXd& operator()(LastLinearLayer& last_lin) { std::cout << "get codes called on last linear layer\n"; return last_lin.layer_output;}    
    Eigen::MatrixXd& operator()(ReluLayer& relu) {  std::cout << "get codes called on relu layer\n";return relu.layer_output;}
    Eigen::MatrixXd& operator()(SigmoidLayer& sigmoid) { std::cout << "get codes called on sigmoid layer\n";return sigmoid.layer_output;}
};

struct CallGetWeights {
    Eigen::MatrixXd& operator()(LinearLayer& lin) { return lin.weight; }    
    Eigen::MatrixXd& operator()(LastLinearLayer& last_lin) { return last_lin.weight;} 
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead   
    Eigen::MatrixXd& operator()(ReluLayer& relu) {  std::cout << "get weights called on relu layer\n";return relu.layer_output;}
    Eigen::MatrixXd& operator()(SigmoidLayer& sigmoid) { std::cout << "get weights called on sigmoid layer\n";return sigmoid.layer_output;}
};

struct CallGetBiases {
    Eigen::VectorXd& operator()(LinearLayer& lin) { return lin.bias; }    
    Eigen::VectorXd& operator()(LastLinearLayer& last_lin) { return last_lin.bias;} 
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead   
    Eigen::VectorXd& operator()(ReluLayer& relu) {  std::cout << "get biases called on relu layer\n";Eigen::VectorXd dummy = Eigen::VectorXd::Ones(1); return dummy;}
    Eigen::VectorXd& operator()(SigmoidLayer& sigmoid) { std::cout << "get biases called on sigmoid layer\n";Eigen::VectorXd dummy = Eigen::VectorXd::Ones(1); return dummy;}
};

struct CallGetDout {
    Eigen::MatrixXd& operator()(LinearLayer& lin) { std::cout << "get dout called on lin layer"; return lin.layer_output; }    
    Eigen::MatrixXd& operator()(LastLinearLayer& last_lin) { std::cout << "get dout called on last lin layer"; return last_lin.layer_output;} 
    Eigen::MatrixXd& operator()(ReluLayer& relu) {  return relu.dout;}
    Eigen::MatrixXd& operator()(SigmoidLayer& sigmoid) { return sigmoid.dout;}
};

std::vector<std::variant<LinearLayer, LastLinearLayer, ReluLayer, SigmoidLayer>> variant_vec;


//Need to init layer outputs properly 
//Worth having a seperate conv network class
class Conv2dLayer{
public:
    std::vector<std::vector<Eigen::MatrixXd>> kernels; 
    std::vector<std::vector<Adam>> adam_optimisers; 
    std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
    std::vector<std::vector<Eigen::MatrixXd>> codes;
    std::vector<std::vector<Eigen::MatrixXd>> dWs;
    std::vector<std::vector<Eigen::MatrixXd>> layer_douts;
    Eigen::VectorXd db; 
    Eigen::VectorXd bias;
    

    int N;
    int C_in;
    int C_out;
    int H;
    int W;
    bool init_vals = true;

    Conv2dLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, Eigen::VectorXd bias, int N, int C_in, int H, int W) : kernels(kernels), bias(bias), N(N), C_in(C_in), C_out(kernels.size()), H(H), W(W){
        for (int n = 0; n < N; n++){
            std::vector<Eigen::MatrixXd> tmp;
            for (int c = 0; c < C_out; c++){
                tmp.emplace_back(Eigen::MatrixXd(H,W));
            }
            layer_outputs.emplace_back(tmp);
            layer_douts.emplace_back(tmp);
        } 
        //THink need to init db proper
        db.setZero(bias.size());
       
        for (int c_out = 0; c_out < C_out; c_out++){
            std::vector<Eigen::MatrixXd> tmp;
            std::vector<Adam> adam_tmp;
            for (int c_in = 0; c_in < C_in; c_in++){
                tmp.emplace_back(Eigen::MatrixXd(kernels[0][0].rows(),kernels[0][0].rows()));
                adam_tmp.emplace_back(Adam(kernels[0][0].rows(), kernels[0][0].rows(), 0.008));
            }
            dWs.emplace_back(tmp);
            adam_optimisers.emplace_back(adam_tmp);
        }
    }


    //bias deffo wrong updated c_out to many times for each bias. Kinda not easy to fix. 
    ALTMIN_INLINE void adam() noexcept {

        

        if (init_vals){
            adam_optimisers[0][0].template adam<true,true>(kernels[0][0], bias, dWs[0][0], db);
            for (int c_out = 0; c_out < C_out; c_out++){
                for (int c_in = 0; c_in < C_in; c_in++){
                    if ((c_out + c_in) == 0){
                        continue;
                    }else{
                        adam_optimisers[c_out][c_in].template adam<true,false>(kernels[c_out][c_in], bias, dWs[c_out][c_in], db);
                    }
                    
                }
       
            }
            
        }else{
            //update bias omce
            adam_optimisers[0][0].template adam<false,true>(kernels[0][0], bias, dWs[0][0], db);
            for (int c_out = 0; c_out < C_out; c_out++){
                for (int c_in = 0; c_in < C_in; c_in++){
                    if ((c_out + c_in) == 0){
                        continue;
                    }else{
                        adam_optimisers[c_out][c_in].template adam<false,false>(kernels[c_out][c_in], bias, dWs[c_out][c_in], db);
                    }
                }
            }
            
        }
        init_vals=false;
    }


    //works for one channel in atm
    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const bool train) noexcept{
        for (int n = 0; n < inputs.size(); n++){
            for (int c_out = 0; c_out < C_out; c_out++){
                Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(H,W);
                //Bias only needs to be added once so do like this so its done during the first convolution so we don't have to iterate over the whole matrix again at the end.
                double bias_val = bias[c_out];
                for (int c_in = 0; c_in < C_in; c_in++){
                    sum +=  conv2d(inputs[n][c_in], kernels[c_out][c_in], bias_val, H, W);
                    bias_val = 0.0;
                }
                
                layer_outputs[n][c_out] = sum;
                
            }
        }
        if (train){ codes = layer_outputs; }      
    }

    template <typename T>
    ALTMIN_INLINE void update_codes(const T &dc) noexcept{
        for (int n = 0; n < N; n++){
            for (int c_out = 0; c_out < C_out; c_out++){
                codes[n][c_out] -= (((1.0 + 0.9) * 0.3) * dc[n][c_out]);
            }
        }
    }


    template <typename T>
    ALTMIN_INLINE void differentiate_layer(const T& inputs, const T& dL_dout) noexcept{
     
    

        for (int c_out = 0; c_out < C_out; c_out++){
            for (int c_in = 0; c_in < C_in; c_in++){
                dWs[c_out][c_in].setZero();
            }
        }


        for (int n = 0; n < N; n++){
            for (int c_out = 0; c_out < C_out; c_out++){
                //Bias only needs to be added once so do like this so its done during the first convolution so we don't have to iterate over the whole matrix again at the end.
                for (int c_in = 0; c_in < C_in; c_in++){
                    dWs[c_out][c_in] +=  differentiate_conv2d(inputs[n][c_in], dL_dout[n][c_out], kernels[0][0].rows(), kernels[0][0].cols(), true);
                }
            }
        }

        //bias


        db.setZero();



        for (int n = 0; n < N; n++){
            for (int c =0; c<C_out; c++){
                db[c] += (dL_dout[n][c].sum());
            }
        }



            

        //layer_dout

    



        for (int n =0 ; n<N ; n++){
            for(int c_in = 0; c_in < C_in; c_in++){

            
                Eigen::MatrixXd sum = differentiate_conv2d(dL_dout[n][0],kernels[0][c_in],  inputs[0][0].rows(), inputs[0][0].cols(), false);
                for (int c_out =1; c_out < C_out; c_out++){
                    sum += differentiate_conv2d(dL_dout[n][c_out],kernels[c_out][c_in],  inputs[0][0].rows(), inputs[0][0].cols(), false);
                }
                layer_douts[n][c_in] = sum;
            }
        }
        


    }


};

class ReluCNNLayer{
public:

    std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
    std::vector<std::vector<Eigen::MatrixXd>> layer_douts;
    int N;
    int C;
    int H;
    int W;
    ReluCNNLayer(const int N, const int C, const int H, const int W) : N(N), C(C), H(H), W(W){
        for (int n = 0; n < N; n++){
            std::vector<Eigen::MatrixXd> tmp;
            for (int c = 0; c < C; c++){
                tmp.emplace_back(Eigen::MatrixXd(H,W));
            }
            layer_outputs.emplace_back(tmp);
            layer_douts.emplace_back(tmp);
        } 
    };
    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs) noexcept{
        for (int n = 0; n < inputs.size(); n++){
            for (int c = 0; c < C; c++){
                layer_outputs[n][c] = ReLU(inputs[n][c]);
            }
        }   
    }

    template <typename T>
    ALTMIN_INLINE void differentiate_layer(const T& inputs) noexcept{
        for (int n = 0; n < N; n++){
            for (int c = 0; c < C; c++){
                layer_douts[n][c] = differentiate_ReLU(inputs[n][c]);
            }
        }   
    }
};

class MaxPool2dLayer{
public: 
    std::vector<std::vector<Eigen::MatrixXd>> layer_outputs;
    std::vector<std::vector<Eigen::MatrixXd>> layer_douts;
    int kernel_size;
    int stride;
    int N;
    int C;
    int H;
    int W;

    MaxPool2dLayer(int kernel_size, int stride, int N, int C, int H, int W) :  kernel_size(kernel_size), stride(stride),  N(N), C(C), H(H), W(W){
        for (int n = 0; n < N; n++){
            std::vector<Eigen::MatrixXd> tmp;
            for (int c = 0; c < C; c++){
                tmp.emplace_back(Eigen::MatrixXd(H, W));
            }
            layer_outputs.emplace_back(tmp);
            layer_douts.emplace_back(tmp);
        } 
    }

    //works for one channel in atm
    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs) noexcept{
        for (int n = 0; n < inputs.size(); n++){
            for (int c = 0; c < C; c++){
                layer_outputs[n][c] = maxpool2d(inputs[n][c], kernel_size, stride, H, W);
            }
        }      
    }

    template <typename T>
    ALTMIN_INLINE void differentiate_layer(const T& inputs) noexcept{
        for (int n = 0; n < N; n++){
            for (int c = 0; c < C; c++){
                layer_douts[n][c] = differentiate_maxpool2d(inputs[n][c], kernel_size, stride, H, W);
            }
        }   
    }
};

// class FlattenLayer{
// public:
//     Eigen::MatrixXd layer_outputs;
//     FlattenLayer() {}
//     template <typename T>
//     ALTMIN_INLINE void forward(const T& inputs) noexcept{
//         layer_outputs = flatten(inputs, inputs.size());
//     }
// };

template <typename T>
struct CallForwardCNN {
    void operator()(Conv2dLayer& conv2d) { conv2d.forward(inputs, train); }    
    void operator()(ReluCNNLayer& relu_cnn) { relu_cnn.forward(inputs); }   
    void operator()(MaxPool2dLayer& maxpool2d) { maxpool2d.forward(inputs); }   
   // void operator()(FlattenLayer& flatten) { flatten.forward(inputs); }   
    const T& inputs;
    const bool train = true;
};

struct CallGetCNNCodes {
    std::vector<std::vector<Eigen::MatrixXd>> operator()(Conv2dLayer& conv2d) { return conv2d.codes; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    std::vector<std::vector<Eigen::MatrixXd>> operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get codes called on relu cnn layer\n"; return relu_cnn.layer_outputs;}   
    std::vector<std::vector<Eigen::MatrixXd>> operator()(MaxPool2dLayer& maxpool2d) {std::cout << "get codes called on maxpool2d layer\n";return maxpool2d.layer_outputs; }   
};

struct CallGetCNNWeights {
    std::vector<std::vector<Eigen::MatrixXd>> operator()(Conv2dLayer& conv2d) { return conv2d.kernels; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    std::vector<std::vector<Eigen::MatrixXd>> operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get get weights called on relu cnn layer\n"; return relu_cnn.layer_outputs;}   
    std::vector<std::vector<Eigen::MatrixXd>> operator()(MaxPool2dLayer& maxpool2d) {std::cout << "get get weights called on maxpool2d layer\n";return maxpool2d.layer_outputs; }   
};

struct CallGetCNNBiases {
    Eigen::VectorXd operator()(Conv2dLayer& conv2d) { return conv2d.bias; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    Eigen::VectorXd  operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get bias called on relu cnn layer\n"; return Eigen::VectorXd(1);}   
    Eigen::VectorXd  operator()(MaxPool2dLayer& maxpool2d) {std::cout << "get bias called on maxpool2d layer\n";return Eigen::VectorXd(1); }   
};



struct CallGetdW {
    std::vector<std::vector<Eigen::MatrixXd>> operator()(Conv2dLayer& conv2d) { return conv2d.dWs; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    std::vector<std::vector<Eigen::MatrixXd>> operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get dw called on relu cnn layer\n"; return relu_cnn.layer_outputs;}   
    std::vector<std::vector<Eigen::MatrixXd>> operator()(MaxPool2dLayer& maxpool2d) {std::cout << "get dw called on maxpool2d layer\n";return maxpool2d.layer_outputs; }   
};

struct CallGetdb {
    Eigen::VectorXd operator()(Conv2dLayer& conv2d) { return conv2d.db; }    
    //IDeally would be void but these functions are never called as the nn checks that the layer is linear before calling get codes
    //Return layer output to make compiler happy but never actually gets called so no overhead
    Eigen::VectorXd  operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get db called on relu cnn layer\n"; return Eigen::VectorXd(1);}   
    Eigen::VectorXd  operator()(MaxPool2dLayer& maxpool2d) {std::cout << "get db called on maxpool2d layer\n";return Eigen::VectorXd(1); }   
};


struct CallGetStride {
    int operator()(Conv2dLayer& conv2d) { std::cout << "get stride called on conv2d cnn layer\n"; return -1; }    
    int operator()(ReluCNNLayer& relu_cnn) {  std::cout << "get stride called on relu cnn layer\n"; return -1;}   
    int operator()(MaxPool2dLayer& maxpool2d) { return maxpool2d.stride; }   
};

template <typename T>
struct CallCNNDifferentiateLayer {
    void operator()(Conv2dLayer& conv2d) { conv2d.differentiate_layer(inputs, dL_douts); }    
    void operator()(ReluCNNLayer& relu_cnn) { relu_cnn.differentiate_layer(inputs); }   
    void operator()(MaxPool2dLayer& maxpool2d) { maxpool2d.differentiate_layer(inputs); }   
    // void operator()(FlattenLayer& flatten) { flatten.forward(inputs); }   
    const T& inputs;
    const T& dL_douts;
};

template <typename T>
struct CallCNNUpdateCodes {
    void operator()(Conv2dLayer& conv2d) { conv2d.update_codes(dc); }    
    void operator()(ReluCNNLayer& relu_cnn) { std::cout << "Update codes called for relu " << std::endl;}   
    void operator()(MaxPool2dLayer& maxpool2d) { std::cout << "update codes called for maxpool2d " << std::endl; }   
    // void operator()(FlattenLayer& flatten) { flatten.forward(inputs); }   
    const T& dc;
};

struct CallAdamCNN {
    void operator()(Conv2dLayer& conv2d) { conv2d.adam(); }    
    void operator()(ReluCNNLayer& relu_cnn) { std::cout << "adam called called for relu " << std::endl;}   
    void operator()(MaxPool2dLayer& maxpool2d) { std::cout << "adam called called for maxpool2d " << std::endl; }   
};


//std::vector<std::vector<Eigen::MatrixXd>>
const auto&  get_layer_output_cnn = [](const auto& layer) {  return layer.layer_outputs;};
const auto&  get_layer_douts_cnn = [](const auto& layer) {  return layer.layer_douts;};












#endif  // FAST_ALTMIN_LAYERS_H