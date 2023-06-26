//
// Created by mbarbone on 6/26/23.
//

#ifndef FAST_ALTMIN_LAYERS_H
#define FAST_ALTMIN_LAYERS_H

#include "defines.h"
#include "functions.hpp"
#include "types.h"

struct Layer {
    const int n;
    const int batch_size;
    Eigen::MatrixXd layer_output;
    Eigen::MatrixXd dout;

    Layer(const int n, const int batch_size) : n(n), batch_size(batch_size), layer_output(batch_size, n) {}
};

class Adam {
   public:
    Adam(Eigen::MatrixXd& weight,
         Eigen::VectorXd& bias,
         const double learning_rate,
         const float beta1 = 0.9,
         const float beta2 = 0.999,
         const float eps   = 1e-08)
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
          bias_v_t_correct(Eigen::VectorXd::Zero(bias.rows())) {}

   private:
    // See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html for
    // implementation details
    template <bool init_vals>
    ALTMIN_INLINE void adam(const nanobind::DRef<Eigen::MatrixXd>& grad_weight,
                            const nanobind::DRef<Eigen::VectorXd>& grad_bias) noexcept {
        // Update weight
        if constexpr (init_vals) {
            weight_m = (1 - beta_1) * grad_weight;
            weight_v = (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        } else {
            weight_m = beta_1 * weight_m + (1 - beta_1) * grad_weight;
            weight_v = beta_2 * weight_v + (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        }
        weight_m_t_correct = weight_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        weight_v_t_correct = weight_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));

        weight =
            weight -
            learning_rate * (weight_m_t_correct.cwiseQuotient((weight_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        // Update bias
        if constexpr (init_vals) {
            bias_m = (1 - beta_1) * grad_bias;
            bias_v = (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
        } else {
            bias_m = beta_1 * bias_m + (1 - beta_1) * grad_bias;
            bias_v = beta_2 * bias_v + (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
        }
        bias_m_t_correct = bias_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        bias_v_t_correct = bias_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));
        bias             = bias -
               learning_rate * (bias_m_t_correct.cwiseQuotient((bias_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        step = step + 1;
    }

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
    const double learning_rate;
    const float beta_1;
    const float beta_2;
    const float eps;
    int step = 1;
};

struct NonLinear : public Layer {
    NonLinear(const int n, const int batchSize) : Layer(n, batchSize) {}
};

struct Linear : public Layer {
    const int m;
    Eigen::MatrixXd m_weight;
    Eigen::VectorXd m_bias;
    Eigen::MatrixXd m_codes;

    Linear(const int n,
           const int batchSize,
           const int m,
           Eigen::MatrixXd weight,
           Eigen::VectorXd bias,
           const double learning_rate)
        : Layer(n, batchSize),
          m(m),
          m_weight(std::move(weight)),
          m_bias(std::move(bias)),
          m_codes(n, batchSize),
          adam(m_weight, m_bias, learning_rate) {}

    ALTMIN_INLINE void forward(const nanobind::DRef<Eigen::MatrixXd>& inputs, const bool store_codes) noexcept {
        layer_output = lin(inputs, m_weight, m_bias);
        if (store_codes) { m_codes = layer_output; }
    }
    ALTMIN_INLINE void differentiate_layer(const nanobind::DRef<Eigen::MatrixXd>& inputs,
                                           bool code_derivative) noexcept {
        if (code_derivative) {
            dout = m_weight;
        } else {
            // Only need one of these
            dout = inputs;
        }
    }

   private:
    Adam adam;
};

// last linear layer does not have codes
struct LastLinear : public Linear {
    LastLinear(const int n,
               const int batchSize,
               const int m,
               const Eigen::MatrixXd& weight,
               const Eigen::VectorXd& bias,
               const double learningRate)
        : Linear(n, batchSize, m, weight, bias, learningRate) {}
};

struct Relu : public NonLinear {
    Relu(const int n, const int batchSize) : NonLinear(n, batchSize) {}
    ALTMIN_INLINE void forward(const nanobind::DRef<Eigen::MatrixXd>& inputs) noexcept { layer_output = ReLU(inputs); }
    ALTMIN_INLINE void differentiate_layer(const nanobind::DRef<Eigen::MatrixXd>& inputs) noexcept {
        dout = differentiate_ReLU(inputs);
    }
};

struct Sigmoid : public NonLinear {
    Sigmoid(const int n, const int batchSize) : NonLinear(n, batchSize) {}
    ALTMIN_INLINE void forward(const nanobind::DRef<Eigen::MatrixXd>& inputs) noexcept {
        layer_output = sigmoid(inputs);
    }
    ALTMIN_INLINE void differentiate_layer(const nanobind::DRef<Eigen::MatrixXd>& inputs) noexcept {
        dout = differentiate_sigmoid(inputs);
    }
};

#endif  // FAST_ALTMIN_LAYERS_H
