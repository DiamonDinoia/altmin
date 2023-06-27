//
// Created by mbarbone on 6/26/23.
//

#ifndef FAST_ALTMIN_NEURAL_NETWORK_H
#define FAST_ALTMIN_NEURAL_NETWORK_H

#include "layers.h"

enum loss_t { BCE, MSE, CrossEntropy };

namespace {
using layer_t = std::variant<Linear, Relu, Sigmoid, LastLinear>;
}

template <loss_t Loss>
class NeuralNetwork {
   public:
    NeuralNetwork(int n_iter_codes,
                  int n_iter_weights,
                  int batch_size,
                  int m,
                  double lr_codes,
                  double mu,
                  double momentum)
        :  ////////// Model hyperparemters
           /////////////////////////////////////////////////////////////////////
          n_iter_codes(n_iter_codes),
          n_iter_weights(n_iter_weights),
          lr_codes(lr_codes),
          mu(mu),
          momentum(momentum),
          batch_size(batch_size),
          //////////////////////////////////////////////////////////////////////
          inputs(batch_size, m) {}
    // Create a non linear layer and add to layers vector

    ALTMIN_INLINE void emplace_relu(const int batch_size, const int n) { layers.emplace_back(Relu(n, batch_size)); }

    ALTMIN_INLINE void emplace_sigmoid(const int batch_size, const int n) {
        layers.emplace_back(Sigmoid{n, batch_size});
    }

    ALTMIN_INLINE void emplace_linear(const int batch_size,
                                      const int n,
                                      const int m,
                                      const nanobind::DRef<Eigen::MatrixXd>& weight,
                                      const nanobind::DRef<Eigen::VectorXd>& bias,
                                      const double lr) {
        layers.emplace_back(Linear{n, batch_size, m, weight, bias, lr});
    }

    ALTMIN_INLINE void emplace_last_linear(const int batch_size,
                                           const int n,
                                           const int m,
                                           const nanobind::DRef<Eigen::MatrixXd>& weight,
                                           const nanobind::DRef<Eigen::VectorXd>& bias,
                                           const double lr) {
        layers.emplace_back(LastLinear{n, batch_size, m, weight, bias, lr});
    }

    ALTMIN_INLINE int get_idx_next_layer_with_codes(int idx) noexcept {
        auto index = 0;
        for (int i = idx; i < layers.size(); ++i) {
            if (std::holds_alternative<Linear>(layers[i])) { return index; }
        }
        return static_cast<int>(layers.size() - 1);
    }

    ALTMIN_INLINE void add_to_vectors(const int start_idx,
                                      const int layer_idx,
                                      const int end_idx,
                                      const int derivative_idx) {
        weight_pairs.emplace_back(start_idx, layer_idx, end_idx, derivative_idx);
        auto getWeight = [this, layer_idx]() {
            auto& arg = layers[layer_idx];
            if (std::holds_alternative<Linear>(arg)) { return std::get<Linear>(arg).m_weight; }
            return std::get<LastLinear>(arg).m_weight;
        };
        const auto& weight = getWeight();
        weight_dL_douts.emplace_back(batch_size, weight.rows());
        dL_dWs.emplace_back(weight.rows(), weight.cols());
        dL_dbs.emplace_back(weight.rows(), 1);
    }

    // Create pairs of layers needed to update the codes and weights so you
    // don't have to iterate over all the layers each time Only run once after
    // all layers have been added to the nn
    ALTMIN_INLINE void construct_pairs() {
        int end_idx;
        int start_idx      = 0;
        int derivative_idx = 0;
        auto last          = false;
        for (auto idx = 0; idx < layers.size() && !last; idx++) {
            std::visit(
                [&idx, &last, &start_idx, &end_idx, &derivative_idx, this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, Linear>) {
                        end_idx = get_idx_next_layer_with_codes(idx);
                        add_to_vectors(start_idx, idx, end_idx, derivative_idx);
                        derivative_idx += 1;
                    }
                    if constexpr (std::is_same_v<T, LastLinear>) {
                        end_idx = get_idx_next_layer_with_codes(idx);
                        code_pairs.insert(code_pairs.begin(), std::make_tuple(idx, end_idx));
                        add_to_vectors(start_idx, idx, idx, derivative_idx);
                        derivative_idx += 1;
                        start_idx = idx;
                        last      = true;
                    }
                    // if it is not a linear layer or a last linear layer it
                    // does nothing
                },
                layers[idx]);
        }
    }

    // //This is used in the calculate the input for the partial derivative
    ALTMIN_INLINE void calc_matrix_for_derivative(const nanobind::DRef<Eigen::MatrixXd>& inputs,
                                                  const int idx,
                                                  const int end_idx) noexcept {
        auto applyForward = [&idx, &end_idx](auto&& arg, auto&& inputs) {
            if (std::holds_alternative<Linear>(arg)) { std::get<Linear>(arg).forward(inputs, false); }
            if (std::holds_alternative<Relu>(arg)) { std::get<Relu>(arg).forward(inputs); }
            if (std::holds_alternative<Sigmoid>(arg)) { std::get<Sigmoid>(arg).forward(inputs); }
            if (std::holds_alternative<LastLinear>(arg)) { std::get<LastLinear>(arg).forward(inputs, false); }
        };

        if (idx < end_idx) { applyForward(layers[idx], inputs); }
        if (idx + 1 < end_idx) { applyForward(layers[idx + 1], getLayer(idx).layer_output); }
        if (idx + 2 < end_idx) { applyForward(layers[idx + 2], getLayer(idx + 1).layer_output); }
    }

    // Apply the chain rule to calculate dw, db or dc depending on bool
    // code_derivative
    ALTMIN_INLINE void apply_chain_rule(const bool code_derivative) noexcept {
        for (int idx = end_idx; idx > start_idx; idx--) {
            std::visit(
                [this, idx](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, Linear> || std::is_same_v<T, LastLinear>) {
                        dL_dout = dL_dout.cwiseProduct(arg.dout);
                    }
                    if constexpr (std::is_same_v<T, Sigmoid>) { dL_dout = dL_dout * arg.dout; }
                    if constexpr (std::is_same_v<T, Relu>) { dL_dout = dL_dout.cwiseProduct(arg.dout); }
                },
                layers[idx]);
        }
        dL_dc = dL_dout;
    }

    ALTMIN_INLINE void apply_chain_rule_weights(const bool code_derivative,
                                                const int start_idx,
                                                const int end_idx,
                                                const int derivative_idx) noexcept {
        for (int idx = end_idx; idx > start_idx; idx--) {
            std::visit(
                [this, derivative_idx, idx](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, Linear> || std::is_same_v<T, LastLinear>) {
                        dL_dWs[derivative_idx] = weight_dL_douts[derivative_idx].transpose() * arg.dout;
                        dL_dbs[derivative_idx] = weight_dL_douts[derivative_idx].colwise().sum();
                    }
                    if constexpr (std::is_same_v<T, Sigmoid>) {
                        weight_dL_douts[derivative_idx] = weight_dL_douts[derivative_idx].cwiseProduct(arg.dout);
                    }
                    if constexpr (std::is_same_v<T, Relu>) {
                        weight_dL_douts[derivative_idx] = weight_dL_douts[derivative_idx].cwiseProduct(arg.dout);
                    }
                },
                layers[idx]);
        }
    }

    // USed to calc dw and db,   or dc depending on code_derivative
    // Calc_matrix_for_derivative uses the layer_output member of the prev layer
    // to store the input for partial derivative
    ALTMIN_INLINE void calculate_gradients(const nanobind::DRef<Eigen::MatrixXd>& inputs,
                                           const bool code_derivative,
                                           const int start_idx,
                                           const int end_idx,
                                           const int derivative_idx) noexcept {
        auto apply_differentiate = [](auto&& arg, auto& inputs, auto code_derivative) {
            if (std::holds_alternative<Linear>(arg)) {
                std::get<Linear>(arg).differentiate_layer(inputs, code_derivative);
            }
            if (std::holds_alternative<Relu>(arg)) { std::get<Relu>(arg).differentiate_layer(inputs); }
            if (std::holds_alternative<Sigmoid>(arg)) { std::get<Sigmoid>(arg).differentiate_layer(inputs); }
            if (std::holds_alternative<LastLinear>(arg)) {
                std::get<LastLinear>(arg).differentiate_layer(inputs, code_derivative);
            }
        };

        auto getLayerOutput = [](auto&& arg, auto& inputs, auto code_derivative) -> auto& {
            using T = std::decay_t<decltype(arg)>;
            std::get<T>(arg).layer_output;
        };

        auto getDout = [](auto&& arg) -> auto& {
            using T = std::decay_t<decltype(arg)>;
            return std::get<T>(arg).dout;
        };

        if (start_idx == end_idx) {
            apply_differentiate(layers[end_idx], inputs, code_derivative);
            dL_dWs[derivative_idx] = weight_dL_douts[derivative_idx].transpose() * getLinear(start_idx).dout;
            dL_dbs[derivative_idx] = weight_dL_douts[derivative_idx].colwise().sum();
            return;
        } else {
            calc_matrix_for_derivative(inputs, start_idx + 1, end_idx);
            apply_differentiate(layers[end_idx], getLayer(end_idx - 1).layer_output, code_derivative);
        }

        if (end_idx - 1 > start_idx) {
            if (end_idx - 1 == start_idx + 1) {
                apply_differentiate(layers[end_idx - 1], inputs, code_derivative);
            } else {
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx - 1);
                apply_differentiate(layers[end_idx - 1], getLayer(end_idx - 2).layer_output, code_derivative);
            }
        }

        if (end_idx - 2 > start_idx) {
            if (end_idx - 2 == start_idx + 1) {
                apply_differentiate(layers[end_idx - 2], inputs, code_derivative);
            } else {
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx - 2);
                apply_differentiate(layers[end_idx - 2], getLayer(end_idx-3).layer_output, code_derivative);
            }
        }

        if (code_derivative) {
            apply_chain_rule(code_derivative);
        } else {
            apply_chain_rule_weights(false, start_idx, end_idx, derivative_idx);
        }
    }
    // Update the codes
    ALTMIN_INLINE void update_codes(const nanobind::DRef<Eigen::MatrixXd>& targets) noexcept {
        bool last_layer = true;
        int idx_last_code;
        // My way of dealing with cross entropy
        // There will be a better way
        Eigen::VectorXd class_labels;
        if constexpr (Loss == loss_t::CrossEntropy) { class_labels = Eigen::VectorXd{targets.reshaped()}; }

        for (auto [start_idx, end_idx] : code_pairs) {
            for (size_t it = 0; it < n_iter_codes; it++) {
                inputs = getLinear(start_idx).m_codes;
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx + 1);
                if (!last_layer) {
                    dL_dout =
                        differentiate_MSELoss(getLinear(idx_last_code).layer_output, getLinear(idx_last_code).m_codes);

                } else {
                    if constexpr (Loss == loss_t::BCE) {
                        dL_dout = (1.0 / mu) * differentiate_BCELoss(getLinear(end_idx).layer_output, targets);
                    }
                    if constexpr (Loss == loss_t::MSE) {
                        dL_dout = (1.0 / mu) * differentiate_MSELoss(getLinear(end_idx).layer_output, targets);
                    }
                    if constexpr (Loss == loss_t::CrossEntropy) {
                        dL_dout = (1.0 / mu) * differentiate_CrossEntropyLoss(
                                                   getLinear(end_idx).layer_output, class_labels,
                                                   static_cast<int>(getLinear(end_idx).layer_output.cols()));
                    }
                }
                last_layer = false;
                calculate_gradients(getLinear(start_idx).m_codes, true, start_idx, end_idx, -1);
                getLinear(start_idx).m_codes = getLinear(start_idx).m_codes - (((1.0 + momentum) * lr_codes) * dL_dc);
                idx_last_code                = start_idx;
            }
        }
    }

    // //data gets changed in place and is used in multiple places so would need
    // to be copied anyway so no point passing a reference.
    ALTMIN_INLINE void update_weights(const nanobind::DRef<Eigen::MatrixXd>& data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd>& targets_nb) noexcept {
#pragma omp parallel for schedule(dynamic) default(none) shared(data_nb, targets_nb, weight_pairs, layers)
        for (int x = 0; x < weight_pairs.size(); x++) {
            if (x == 0) {
                update_weights_parallel(data_nb, weight_pairs[x], targets_nb, true);
            } else {
                update_weights_parallel(getLinear(std::get<0>(weight_pairs[x])).m_codes, weight_pairs[x], targets_nb,
                                        false);
            }
        }
    }

    ALTMIN_INLINE void update_weights_parallel(const nanobind::DRef<Eigen::MatrixXd>& inputs,
                                               const std::tuple<int, int, int, int>& indexes,
                                               const nanobind::DRef<Eigen::MatrixXd>& targets,
                                               bool first_layer) noexcept {
        int start_idx_parallel      = std::get<0>(indexes);
        int layer_idx_parallel      = std::get<1>(indexes);
        int end_idx_parallel        = std::get<2>(indexes);
        int derivative_idx_parallel = std::get<3>(indexes);
        Eigen::VectorXd class_labels;
        if constexpr (Loss == loss_t::CrossEntropy) { class_labels = Eigen::VectorXd{targets.reshaped()}; }

        for (size_t it = 0; it < n_iter_weights; it++) {
            // populate outputs
            if (first_layer) {
                // weight_inputs[derivative_idx] = data;
                calc_matrix_for_derivative(inputs, start_idx_parallel, end_idx_parallel + 1);

                weight_dL_douts[derivative_idx_parallel] = differentiate_MSELoss(
                    getLinear(layer_idx_parallel).layer_output, getLinear(layer_idx_parallel).m_codes);
            } else {
                calc_matrix_for_derivative(inputs, start_idx_parallel + 1, end_idx_parallel + 1);

                if (end_idx_parallel == (layers.size() - 1)) {
                    if constexpr (Loss == loss_t::BCE) {
                        weight_dL_douts[derivative_idx_parallel] =
                            differentiate_BCELoss(getLinear(end_idx_parallel).layer_output, targets);
                    }
                    if constexpr (Loss == loss_t::MSE) {
                        weight_dL_douts[derivative_idx_parallel] =
                            differentiate_MSELoss(getLinear(end_idx_parallel).layer_output, targets);
                    }
                    if constexpr (Loss == loss_t::CrossEntropy) {
                        weight_dL_douts[derivative_idx_parallel] = differentiate_CrossEntropyLoss(
                            getLinear(end_idx_parallel).layer_output, class_labels,
                            static_cast<int>(getLinear(end_idx_parallel).layer_output.cols()));
                    }

                } else {
                    weight_dL_douts[derivative_idx_parallel] = differentiate_MSELoss(
                        getLinear(end_idx_parallel).layer_output, getLinear(end_idx_parallel).m_codes);
                }
            }

            // Seperate functions so only have to pass the data if necesseray
            // std::optional may be better

            calculate_gradients(inputs, false, start_idx_parallel, end_idx_parallel, derivative_idx_parallel);

            getLinear(layer_idx_parallel).template adam<false>(dL_dWs[derivative_idx_parallel], dL_dbs[derivative_idx_parallel]);
        }
    }

   private:
    std::vector<layer_t> layers;
    Eigen::MatrixXd inputs;

    // Model hyperparams
    size_t n_iter_codes;
    size_t n_iter_weights;
    double lr_codes;
    double lr_weights;
    double mu;
    int criterion;
    double momentum = 0.9;
    int batch_size;

    // Used in parallel weight gradient calc
    std::vector<Eigen::MatrixXd> weight_dL_douts;
    std::vector<Eigen::MatrixXd> dL_dWs;
    std::vector<Eigen::VectorXd> dL_dbs;

    // Used to calc dL_dc
    Eigen::MatrixXd dL_dout;
    Eigen::MatrixXd dL_dc;

    // Used so don't have to iterate over all layers each time to find layers
    // needed to update weights and codes
    std::vector<std::tuple<int, int>> code_pairs;
    std::vector<std::tuple<int, int, int, int>> weight_pairs;
    int start_idx;
    int layer_idx;
    int end_idx;

    ALTMIN_INLINE constexpr Linear& getLinear(int idx) {
        auto& arg = layers[idx];
        if (std::holds_alternative<Linear>(arg)) { return std::get<Linear>(arg); }
        if (std::holds_alternative<Relu>(arg)) {
            throw std::runtime_error("Relu layer not supported");
            return reinterpret_cast<Linear&>(std::get<Relu>(arg));
        }
        if (std::holds_alternative<Sigmoid>(arg)) {
            throw std::runtime_error("Relu layer not supported");
            return reinterpret_cast<Linear&>(std::get<Sigmoid>(arg));
        }
        if (std::holds_alternative<LastLinear>(arg)) { return std::get<LastLinear>(arg); }
    }

    ALTMIN_INLINE constexpr Layer& getLayer(int idx) {
        auto& arg = layers[idx];
        if (std::holds_alternative<Linear>(arg)) { return std::get<Linear>(arg); }
        if (std::holds_alternative<Relu>(arg)) { return std::get<Relu>(arg); }
        if (std::holds_alternative<Sigmoid>(arg)) { return std::get<Sigmoid>(arg); }
        if (std::holds_alternative<LastLinear>(arg)) { return std::get<LastLinear>(arg); }
    }
};

#endif  // FAST_ALTMIN_NEURAL_NETWORK_H
