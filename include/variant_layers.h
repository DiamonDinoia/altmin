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

    template <bool init_vals, typename T, typename G>
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
   

        // Update bias
        if constexpr(init_vals){
            bias_m = (1 - beta_1) * grad_bias;
            bias_v = (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
        } else {
            bias_m = beta_1 * bias_m + (1 - beta_1) * grad_bias;
            bias_v = beta_2 * bias_v + (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
        }

   
        bias_m_t_correct = bias_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        bias_v_t_correct = bias_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));
   
        

        bias -= learning_rate * (bias_m_t_correct.cwiseQuotient((bias_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        step = step + 1;
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
    double learning_rate;
    Adam adam_optimiser;
    bool init_vals = true;

    LinearLayer(const int batchSize, Eigen::MatrixXd weight, Eigen::VectorXd bias, const double learning_rate) :
        Layer(layer_type::LINEAR, weight.rows(), batchSize), weight(weight), bias(bias), codes(batch_size,weight.cols()), learning_rate(learning_rate), 
        adam_optimiser(weight.rows(), weight.cols(), learning_rate) {}

    template <typename T>
    ALTMIN_INLINE void forward(const T& inputs, const bool store_codes) noexcept{
        layer_output = lin(inputs, weight, bias);
        if (store_codes){
            codes = layer_output;
        }
        
    }
   
    template <typename T>
    ALTMIN_INLINE void update_codes(const T &dc) noexcept{
        codes -= (((1.0 + 0.9) * 0.3) * dc);
    }

    ALTMIN_INLINE void set_codes(const nanobind::DRef<Eigen::MatrixXd> &codes) noexcept{
        this->codes = codes;
    };


    template <typename T, typename G>
    ALTMIN_INLINE void adam(const T& grad_weight, const G& grad_bias) noexcept {
        if (init_vals){
            adam_optimiser.template adam<true>(weight, bias, grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false>(weight, bias, grad_weight, grad_bias);
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
            adam_optimiser.template adam<true>(weight, bias, grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false>(weight, bias, grad_weight, grad_bias);
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
// variant_vec.emplace_back(LinearLayer{});
// variant_vec.emplace_back(ReLULayer{});
// // variant_vec.emplace_back(One{});
// Eigen::MatrixXd a;
// Eigen::MatrixXd b = Eigen::Ones(2,2);
// for (auto &var: variant_vec){

//     std::cout << std::visit(CallForward<decltype(b)>{b}, var) << std::endl;

// }

//  ALTMIN_INLINE virtual void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs, bool store_codes) noexcept =0;
//     ALTMIN_INLINE virtual void differentiate_layer(const Eigen::Ref<Eigen::MatrixXd>& inputs)noexcept {std::cout << "diff layer called bad " << std::endl;}
//     ALTMIN_INLINE virtual void update_codes(const Eigen::MatrixXd &dc)noexcept {}
//     ALTMIN_INLINE virtual Eigen::MatrixXd& get_codes()noexcept {std::cout << "get_codes called bad " << std::endl;}
//     ALTMIN_INLINE virtual Eigen::MatrixXd& get_weight()noexcept {std::cout << "get weights called bad " << std::endl;}
//     ALTMIN_INLINE virtual Eigen::VectorXd& get_bias()noexcept {std::cout << "get weights called bad " << std::endl;}
//     ALTMIN_INLINE virtual void set_codes(const nanobind::DRef<Eigen::MatrixXd> &codes)noexcept {};
//     ALTMIN_INLINE virtual void adam(const Eigen::Ref<Eigen::MatrixXd>& grad_weight, const Eigen::Ref<Eigen::VectorXd>& grad_bias)noexcept {std::cout << "adam called bad " << std::endl;}















#endif  // FAST_ALTMIN_LAYERS_H