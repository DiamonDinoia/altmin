//
// Created by mbarbone on 6/26/23.
//

//
// Edited by Tom 6/07/23
// I kept the adam class but all the layer class are new  

#ifndef FAST_ALTMIN_LAYERS_H
#define FAST_ALTMIN_LAYERS_H

#include "defines.h"
#include "functions.hpp"
#include "types.h"
#include <thread>

class Adam {
   public:
    Adam(auto& weight,
         auto& bias,
         const double learning_rate,
         float beta1 = 0.9,
         float beta2 = 0.999,
         float eps   = 1e-08)
        : learning_rate(learning_rate),
          weight(weight),
          bias(bias),
          beta_1(beta1),
          beta_2(beta2),
          eps(eps),
          weight_m(Eigen::MatrixXd::Zero(weight.rows(), weight.cols())),
          weight_v(Eigen::MatrixXd::Zero(weight.rows(), weight.cols())),
          bias_m(Eigen::VectorXd::Zero(bias.rows())),
          bias_v(Eigen::VectorXd::Zero(bias.rows())),
          weight_m_t_correct(Eigen::MatrixXd::Zero(weight.rows(), weight.cols())),
          weight_v_t_correct(Eigen::MatrixXd::Zero(weight.rows(), weight.cols())),
          bias_m_t_correct(Eigen::VectorXd::Zero(bias.rows())),
          bias_v_t_correct(Eigen::VectorXd::Zero(bias.rows())){const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);}

    template<bool init_vals>
    ALTMIN_INLINE void adam(const auto& grad_weight,
                            const auto& grad_bias) noexcept {
        
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


    Eigen::MatrixXd& weight;
    Eigen::VectorXd& bias;
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
    Eigen::MatrixXd dout;
    int n;
    int batch_size;
    Layer(layer_type layer, const int n, const int batch_size) : layer(layer), n(n), batch_size(batch_size), layer_output(batch_size, n), dout(batch_size, n) {const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);}
    
    ALTMIN_INLINE virtual void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs, bool store_codes) noexcept =0;
    ALTMIN_INLINE virtual void differentiate_layer(const Eigen::Ref<Eigen::MatrixXd>& inputs)noexcept {std::cout << "diff layer called bad " << std::endl;}
    ALTMIN_INLINE virtual void update_codes(const Eigen::MatrixXd &dc)noexcept {}
    ALTMIN_INLINE virtual Eigen::MatrixXd& get_codes()noexcept {std::cout << "get_codes called bad " << std::endl;}
    ALTMIN_INLINE virtual Eigen::MatrixXd& get_weight()noexcept {std::cout << "get weights called bad " << std::endl;}
    ALTMIN_INLINE virtual Eigen::VectorXd& get_bias()noexcept {std::cout << "get weights called bad " << std::endl;}
    ALTMIN_INLINE virtual void set_codes(const nanobind::DRef<Eigen::MatrixXd> &codes)noexcept {};
    ALTMIN_INLINE virtual void adam(const Eigen::Ref<Eigen::MatrixXd>& grad_weight, const Eigen::Ref<Eigen::VectorXd>& grad_bias)noexcept {std::cout << "adam called bad " << std::endl;}


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
        adam_optimiser(this->weight, this->bias, learning_rate) {}

    
    ALTMIN_INLINE void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs, bool store_codes) noexcept{
        layer_output = lin(inputs, weight, bias);
        if (store_codes){
            codes = layer_output;
        }
        
    }
   

    ALTMIN_INLINE Eigen::MatrixXd& get_codes() noexcept{
        return codes;
    }

    ALTMIN_INLINE Eigen::MatrixXd& get_weight() noexcept{
        return weight;
    }

    ALTMIN_INLINE Eigen::VectorXd& get_bias() noexcept{
        return bias;
    }

    ALTMIN_INLINE void update_codes(const Eigen::MatrixXd &dc) noexcept{
        codes -= (((1.0 + 0.9) * 0.3) * dc);
    }

    ALTMIN_INLINE void set_codes(const nanobind::DRef<Eigen::MatrixXd> &codes) noexcept{
        this->codes = codes;
    };

    ALTMIN_INLINE void adam(const Eigen::Ref<Eigen::MatrixXd>& grad_weight, const Eigen::Ref<Eigen::VectorXd>& grad_bias) noexcept {
        if (init_vals){
            adam_optimiser.template adam<true>(grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false>(grad_weight, grad_bias);
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
        adam_optimiser(this->weight, this->bias, learning_rate) {}

    ALTMIN_INLINE void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs, bool store_codes=false) noexcept{
        layer_output = lin(inputs, weight, bias);
    }

   
    ALTMIN_INLINE Eigen::MatrixXd& get_weight() noexcept{
        return weight;
    }

    ALTMIN_INLINE Eigen::VectorXd& get_bias() noexcept{
        return bias;
    }

    ALTMIN_INLINE void adam(const Eigen::Ref<Eigen::MatrixXd>& grad_weight, const Eigen::Ref<Eigen::VectorXd>& grad_bias) noexcept {
        if (init_vals){
            adam_optimiser.template adam<true>(grad_weight, grad_bias);
        }else{
            adam_optimiser.template adam<false>(grad_weight, grad_bias);
        }
        
        init_vals = false;
    }

};

class ReluLayer : public Layer{
public:
    ReluLayer(const int n, const int batch_size): Layer(layer_type::RELU, n,batch_size){};
    ALTMIN_INLINE void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs, bool store_codes=false) noexcept{
        layer_output = ReLU(inputs);
    }
    ALTMIN_INLINE void differentiate_layer(const Eigen::Ref<Eigen::MatrixXd>& inputs) noexcept{
        dout = differentiate_ReLU(inputs);
    }
};

class SigmoidLayer : public Layer{
public:
    SigmoidLayer(const int n, const int batch_size): Layer(layer_type::SIGMOID, n,batch_size){};
    ALTMIN_INLINE void forward(const Eigen::Ref<Eigen::MatrixXd>& inputs , bool store_codes=false) noexcept{
        layer_output = sigmoid(inputs);
    }
    ALTMIN_INLINE void differentiate_layer(const Eigen::Ref<Eigen::MatrixXd>& inputs) noexcept{
        dout = differentiate_sigmoid(inputs);
    }
};

#endif  // FAST_ALTMIN_LAYERS_H
