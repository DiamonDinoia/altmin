#include "fast_altmin.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <tuple>

// Hello world functions used to test basic input and output for nanobind
int hello_world() {
    std::cout << "c++: Hello World";
    return 0;
}

std::string hello_world_out() { return "Hi python from c++"; }

int hello_world_in(const std::string &av) {
    std::cout << av;
    return 0;
}

// Matrix functions used to test eigen and nanobind
// At the moment the code uses torch instead of eigen but I'm keeping these for
// now as I would be good to transition back to eigen
void matrix_in(const nanobind::DRef<Eigen::MatrixXd> &x) {
    std::cout << x << std::endl;
}

Eigen::MatrixXd matrix_out() {
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2;
    m(0, 1) = -1;
    m(1, 1) = 4;
    return m;
}

Eigen::MatrixXd matrix_multiplication(
        const nanobind::DRef<Eigen::MatrixXd> &n,
        const nanobind::DRef<Eigen::MatrixXd> &m) {
    Eigen::MatrixXd x = n * m;
    return x;
}

class Layer {
public:
    enum layer_type {
        RELU, LINEAR, SIGMOID
    };

    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    Eigen::MatrixXd codes;
    Eigen::MatrixXd layer_output;
    layer_type layer;
    bool has_codes;
    Eigen::MatrixXd dout;
    //Eigen::MatrixXd dL_dc;
    Eigen::MatrixXd dout_dW;
    int dout_db = false;

    // Constructor for linear layers
    Layer(const layer_type layer_type, const int batch_size, const int n, const int m, Eigen::MatrixXd weight,
          Eigen::VectorXd bias, const bool has_codes, const double lr) :
    //////////  Public members accessed by the neural network //////////////////////////////////////////////////////////
            weight(std::move(weight)),
            bias(std::move(bias)),
            codes(batch_size, n),
            layer_output(batch_size, n),
            layer(LINEAR),
            has_codes(has_codes),
            //////////  Used in adam ///////////////////////////////////////////////////////////////////////////////////
            // first and second moments of the gradient of the weight
            weight_m(Eigen::MatrixXd::Zero(n, m)),
            // the exponentially decaying average of the weight gradients
            weight_v(Eigen::MatrixXd::Zero(n, m)),
            // Same as above but for the bias gradient
            weight_m_t_correct(Eigen::MatrixXd::Zero(n, m)),
            weight_v_t_correct(Eigen::MatrixXd::Zero(n, m)),
            bias_m(Eigen::VectorXd::Zero(m)),
            bias_v(Eigen::VectorXd::Zero(m)),
            // Paremeters used by adam
            //Used to initialise the moments of the gradient and then set to false
            init_vals(true),
            lr(lr),
            // Hyperparemters the control the decay rates of the moment estimates
            beta_1(0.9),
            beta_2(0.999),
            eps(1e-08),
            step(1) {}

    // Constructor for non linear layers
    Layer(const layer_type type, const int batch_size, const int n, const double lr) :
            layer(type), has_codes(false),
            lr(lr),
            layer_output(batch_size, n) {}

    // Low priority: Change so returns a string
    void print_info() {
        switch (layer) {
            case (Layer::layer_type::RELU):
                std::cout << "ReLU" << std::endl;
                std::cout << "layer out: (" << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
                break;
            case (Layer::layer_type::LINEAR):
                std::cout << "Lin: ( " << weight.rows() << " , " << weight.cols() << ") has_codes: " << has_codes
                          << std::endl;
                std::cout << "layer out: (" << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
                break;
            case (Layer::layer_type::SIGMOID):
                std::cout << "Sigmoid: " << std::endl;
                std::cout << "layer out: (" << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
                break;
        }
    }

    //
    const Eigen::MatrixXd &get_codes_for_layer() const {
        return codes;
    }

    ALTMIN_INLINE void forward(const Eigen::Ref<Eigen::MatrixXd> &inputs, const bool store_codes) noexcept {
        switch (layer) {
            case (Layer::layer_type::RELU):
                layer_output = inputs;
                ReLU_inplace(layer_output);
                break;

            case (Layer::layer_type::LINEAR):
                layer_output = lin(inputs, weight, bias);
                if (has_codes && store_codes) {
                    codes = layer_output;
                }
                break;

            case (Layer::layer_type::SIGMOID):
                layer_output = inputs;
                sigmoid_inplace(layer_output);
                break;

        }
    }

    // See https://pytorch.org/docs/stable/generated/torch.optim.Adam.html for implementation details
    ALTMIN_INLINE void
    adam(const Eigen::Ref<Eigen::MatrixXd> &grad_weight, const Eigen::Ref<Eigen::VectorXd> &grad_bias) noexcept {

        // Update weight
        if (init_vals) {
            weight_m = (1 - beta_1) * grad_weight;
            weight_v = (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        } else {
            weight_m = beta_1 * weight_m + (1 - beta_1) * grad_weight;
            weight_v = beta_2 * weight_v + (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
        }
        weight_m_t_correct = weight_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        weight_v_t_correct = weight_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));

        weight = weight - lr * (weight_m_t_correct.cwiseQuotient(
                (weight_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        // Update bias
        if (init_vals) {
            bias_m = (1 - beta_1) * grad_bias;
            bias_v = (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
            init_vals = false;
        } else {
            bias_m = beta_1 * bias_m + (1 - beta_1) * grad_bias;
            bias_v = beta_2 * bias_v + (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
        }
        bias_m_t_correct = bias_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
        bias_v_t_correct = bias_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));
        bias = bias - lr * (bias_m_t_correct.cwiseQuotient(
                (bias_v_t_correct.cwiseSqrt().array() + eps).matrix()));

        step = step + 1;
    }


    ALTMIN_INLINE void differentiate_layer(const Eigen::Ref<Eigen::MatrixXd> &inputs, bool code_derivative) noexcept {
        switch (layer) {
            case (Layer::layer_type::RELU):
                dout = differentiate_ReLU(inputs);
                break;
            case (Layer::layer_type::LINEAR):
                if (code_derivative) {
                    dout = weight;
                } else {
                    //Only need one of these
                    dout = inputs;
                    dout_dW = inputs;
                    //don't need this but don't have time to fix
                    dout_db = 1;
                }
                break;
            case (Layer::layer_type::SIGMOID):
                dout = differentiate_sigmoid(inputs);
                break;
        }
    }


private:
    //Used in adam -- see constructor for full details
    Eigen::MatrixXd weight_m;
    Eigen::MatrixXd weight_v;
    Eigen::VectorXd bias_m;
    Eigen::VectorXd bias_v;
    Eigen::MatrixXd weight_m_t_correct;
    Eigen::MatrixXd weight_v_t_correct;
    Eigen::VectorXd bias_m_t_correct;
    Eigen::VectorXd bias_v_t_correct;
    bool init_vals;
    float lr;
    float beta_1;
    float beta_2;
    float eps;
    int step;


};


class NeuralNetwork {
public:

    enum loss_function {
        BCELoss, MSELoss, CrossEntropyLoss
    };

    NeuralNetwork(const loss_function loss_fn, int n_iter_codes, int n_iter_weights, int batch_size, int m,
                  double lr_codes,
                  double mu, double momentum) :
    ////////// Model hyperparemters ////////////////////////////////////////////////////////////////////////////////////
            n_iter_codes(n_iter_codes), n_iter_weights(n_iter_weights), lr_codes(lr_codes), mu(mu), momentum(momentum),
            loss_fn(loss_fn),
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            inputs(batch_size, m) {}

    // Create a non linear layer and add to layers vector
    ALTMIN_INLINE void
    push_back_non_lin_layer(const Layer::layer_type layer_type, const int batch_size, const int n, const double lr) {
        //Have to be shared not unique as python needs access to them as well
        //https://github.com/pybind/pybind11/issues/115
        //maybe nanobind has fixed this by now but I couldn't make it work
        layers.emplace_back(layer_type, batch_size, n, lr);
    }

    // Create a linear layer and add to layers vector
    ALTMIN_INLINE void push_back_lin_layer(Layer::layer_type layer_type, const int batch_size, const int n, const int m,
                                           nanobind::DRef<Eigen::MatrixXd> weight, nanobind::DRef<Eigen::VectorXd> bias,
                                           const bool has_codes, const double lr) {
        layers.emplace_back(layer_type, batch_size, n, m, weight, bias, has_codes, lr);
    }

    // Helper method used in construct pairs
    int get_idx_next_layer_with_codes(int idx) {
        for (int index = idx + 1; index < layers.size(); index++) {
            if (layers[index].has_codes) {
                return index;
            }
        }
        return static_cast<int>(layers.size() - 1);
    }

    // Create pairs of layers needed to update the codes and weights so you don't have to iterate over all the layers each time
    // Only run once after all layers have been added to the nn
    ALTMIN_INLINE void construct_pairs() {
        int end_idx;
        int start_idx = 0;
        for (int idx = 0; idx < layers.size(); idx++) {
            switch (layers[idx].layer) {
                case (Layer::layer_type::LINEAR):
                    if (!layers[idx].has_codes) {
                        weight_pairs.emplace_back(start_idx, idx, get_idx_next_layer_with_codes(idx));
                        continue;
                    }
                    end_idx = get_idx_next_layer_with_codes(idx);
                    code_pairs.insert(code_pairs.begin(), std::make_tuple(idx, end_idx));
                    weight_pairs.emplace_back(start_idx, idx, idx);
                    start_idx = idx;
                    break;
                default:
                    continue;
            }
        }
    }

    //Low priority but should return a string
    void print_info() {
        for (auto &layer: layers) {
            layer.print_info();
        }
    }

    // Forward pass of model
    // If update_codes=false then model is making predictions and not training and thus don't update the codes
    ALTMIN_INLINE Eigen::MatrixXd
    get_codes(nanobind::DRef<Eigen::MatrixXd> inputs_nb, const bool update_codes) noexcept {
        inputs = inputs_nb;
        layers[0].forward(inputs, update_codes);
        for (int idx = 1; idx < layers.size(); idx++) {
            layers[idx].forward(layers[idx - 1].layer_output, update_codes);
        }
        return layers[layers.size() - 1].layer_output;
    }

    // TODO: I Need to change //////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Eigen::MatrixXd> return_codes() {
        std::vector<Eigen::MatrixXd> codes;
        for (const auto &layer: layers) {
            if (layer.has_codes) {
                codes.emplace_back(layer.codes);
            }
        }
        return codes;
    }

    void set_codes(std::vector<nanobind::DRef<Eigen::MatrixXd>> codes) {
        int y = 0;
        for (auto &layer: layers) {
            if (layer.has_codes) {
                layer.codes = codes[y];
                y += 1;
            }

        }
    }

    std::vector<Eigen::MatrixXd> return_weights() {
        std::vector<Eigen::MatrixXd> weights_vec;

        for (auto &layer: layers) {
            switch (layer.layer) {
                case (Layer::layer_type::LINEAR):
                    weights_vec.emplace_back(layer.weight);
                    break;
                default:
                    continue;
            }
        }
        return weights_vec;
    }

    std::vector<Eigen::VectorXd> return_bias() {
        std::vector<Eigen::VectorXd> bias_vec;

        for (auto &layer: layers) {
            switch (layer.layer) {
                case (Layer::layer_type::LINEAR):
                    bias_vec.emplace_back(layer.bias);
                    break;
                default:
                    continue;
            }
        }
        return bias_vec;
    }

    void set_weights_and_biases(std::vector<nanobind::DRef<Eigen::MatrixXd>> weights,
                                std::vector<nanobind::DRef<Eigen::VectorXd>> biases) {
        int y = 0;
        for (auto &layer: layers) {
            switch (layer.layer) {
                case (Layer::layer_type::LINEAR):
                    layer.weight = weights[y];
                    layer.bias = biases[y];
                    y += 1;
                    break;
                default:
                    continue;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // //This is used in the calculate the input for the partial derivative
    ALTMIN_INLINE void calc_matrix_for_derivative(const Eigen::Ref<Eigen::MatrixXd> &inputs, const int idx,
                                                  const int end_idx) noexcept {
        if (idx < end_idx) {
            layers[idx].forward(inputs, false);
        }

        if (idx + 1 < end_idx) {
            layers[idx + 1].forward(layers[idx].layer_output, false);
        }

        if (idx + 2 < end_idx) {
            layers[idx + 2].forward(layers[idx + 1].layer_output, false);
        }

    }

    //Apply the chain rule to calculate dw, db or dc depending on bool code_derivative
    void apply_chain_rule(const bool code_derivative) {
        for (int idx = end_idx; idx > start_idx; idx--) {
            switch (layers[idx].layer) {
                case (Layer::layer_type::RELU):
                    dL_dout = dL_dout.cwiseProduct(layers[idx].dout);
                    break;
                case (Layer::layer_type::LINEAR):
                    if (code_derivative) {
                        dL_dout = dL_dout * layers[idx].dout;
                    } else {
                        dL_dW = dL_dout.transpose() * layers[idx].dout;
                        dL_db = dL_dout.colwise().sum();
                        return;
                    }
                    break;
                case (Layer::layer_type::SIGMOID):
                    dL_dout = dL_dout.cwiseProduct(layers[idx].dout);
                    break;

            }

        }
        dL_dc = dL_dout;
    }

    //Calculate grad for first layer for weights
    //This is probably mergable with calculate_gradient_first_layer I meant to do this but didn't have time
    ALTMIN_INLINE void calculate_gradients_first_layer(const Eigen::MatrixXd &data) noexcept {
        if (start_idx == end_idx) {
            inputs = data;
            layers[end_idx].differentiate_layer(inputs, false);
        } else {
            calc_matrix_for_derivative(inputs, start_idx, end_idx);
            layers[end_idx].differentiate_layer(layers[end_idx - 1].layer_output, false);
        }

        if (end_idx - 1 > start_idx) {
            inputs = data;
            calc_matrix_for_derivative(inputs, start_idx, end_idx - 1);
            layers[end_idx - 1].differentiate_layer(layers[end_idx - 2].layer_output, false);
        }

        if (end_idx - 2 > start_idx) {
            inputs = data;
            calc_matrix_for_derivative(inputs, start_idx, end_idx - 2);
            layers[end_idx - 2].differentiate_layer(layers[end_idx - 3].layer_output, false);
        }

        dL_dW = dL_dout.transpose() * layers[start_idx].dout_dW;
        dL_db = dL_dout.colwise().sum();

    }

    //USed to calc dw and db,   or dc depending on code_derivative
    //Calc_matrix_for_derivative uses the layer_output member of the prev layer to store the input for partial derivative
    ALTMIN_INLINE void calculate_gradients(const bool code_derivative) noexcept {
        if (start_idx == end_idx) {
            inputs = layers[start_idx].codes;
            layers[end_idx].differentiate_layer(inputs, code_derivative);
        } else {
            calc_matrix_for_derivative(inputs, start_idx + 1, end_idx);
            layers[end_idx].differentiate_layer(layers[end_idx - 1].layer_output, code_derivative);
        }

        if (end_idx - 1 > start_idx) {
            if (end_idx - 1 == start_idx + 1) {
                inputs = layers[start_idx].codes;
                layers[end_idx - 1].differentiate_layer(inputs, code_derivative);
            } else {
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx - 1);
                layers[end_idx - 1].differentiate_layer(layers[end_idx - 2].layer_output, code_derivative);
            }
        }

        if (end_idx - 2 > start_idx) {

            if (end_idx - 2 == start_idx + 1) {
                inputs = layers[start_idx].codes;
                layers[end_idx - 2].differentiate_layer(inputs, code_derivative);
            } else {
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx - 2);
                layers[end_idx - 2].differentiate_layer(layers[end_idx - 3].layer_output, code_derivative);
            }

        }

        apply_chain_rule(code_derivative);

    }


    //Update the codes
    ALTMIN_INLINE void update_codes(nanobind::DRef<Eigen::MatrixXd> targets) noexcept {
        bool last_layer = true;
        int idx_last_code;

        //My way of dealing with cross entropy
        //There will be a better way
        Eigen::VectorXd class_labels;
        if (loss_fn == NeuralNetwork::loss_function::CrossEntropyLoss) {
            class_labels = Eigen::VectorXd{targets.reshaped()};
        }

        for (auto indexes: code_pairs) {
            start_idx = std::get<0>(indexes);
            end_idx = std::get<1>(indexes);
            for (size_t it = 0; it < n_iter_codes; it++) {
                //reset to 0
                // FIXME: is this needed?
                dL_dout.setZero();
                dL_dc.setZero();
                //Use the code to predic the next code or model output if next layer
                inputs = layers[start_idx].codes;
                calc_matrix_for_derivative(inputs, start_idx + 1, end_idx + 1);

                if (last_layer) {
                    switch (loss_fn) {
                        case (NeuralNetwork::loss_function::BCELoss):
                            dL_dout = (1.0 / mu) * differentiate_BCELoss(layers[end_idx].layer_output, targets);
                            break;
                        case (NeuralNetwork::loss_function::MSELoss):
                            dL_dout = (1.0 / mu) * differentiate_MSELoss(layers[end_idx].layer_output, targets);
                            break;
                        case (NeuralNetwork::loss_function::CrossEntropyLoss):
                            dL_dout = (1.0 / mu) * differentiate_CrossEntropyLoss(
                                    layers[end_idx].layer_output, class_labels,
                                    static_cast<int>(layers[end_idx].layer_output.cols()));
                            break;
                        default:
                            std::cout << "Loss not imp yet" << std::endl;
                            break;
                    }
                } else {
                    dL_dout = differentiate_MSELoss(layers[idx_last_code].layer_output, layers[idx_last_code].codes);
                }
                last_layer = false;
                calculate_gradients(true);
                layers[start_idx].codes = layers[start_idx].codes - (((1.0 + momentum) * lr_codes) * dL_dc);
                idx_last_code = start_idx;
            }
        }
    }

    // //data gets changed in place and is used in multiple places so would need to be copied anyway so no point passing a reference.
    ALTMIN_INLINE void update_weights(const nanobind::DRef<Eigen::MatrixXd> &data,
                                      nanobind::DRef<Eigen::MatrixXd> targets) noexcept {
        bool first_layer = true;
        int layer_idx;
        Eigen::VectorXd class_labels;
        if (loss_fn == NeuralNetwork::loss_function::CrossEntropyLoss) {
            class_labels = Eigen::VectorXd{targets.reshaped()};
        }

        //Can use the weight pairs to do the code inside the loop in parallel
        // FIXME: this cannot be parallelised as is because inputs, dL_dout, dL_dW, dL_db are shared between threads
        //  moreover is first layer is true and applied to all layers in parallel first layer should be set inside
        //  the loop using indexes inputs, dL_dout, dL_dW, dL_db can be copied.
        //  Am I missing something that should be copied too?
        for (auto indexes: weight_pairs) {
            start_idx = std::get<0>(indexes);
            layer_idx = std::get<1>(indexes);
            end_idx = std::get<2>(indexes);
            for (size_t it = 0; it < n_iter_weights; it++) {
                //reset to 0
                dL_dout.setZero();
                dL_dW.setZero();
                dL_db.setZero();

                // populate outputs
                if (first_layer) {
                    inputs = data;
                    calc_matrix_for_derivative(inputs, start_idx, end_idx + 1);
                    dL_dout = differentiate_MSELoss(layers[layer_idx].layer_output, layers[layer_idx].codes);

                } else {
                    inputs = layers[start_idx].codes;

                    calc_matrix_for_derivative(inputs, start_idx + 1, end_idx + 1);
                    if (end_idx == (layers.size() - 1)) {
                        //dL_dout =  differentiate_BCELoss(outputs, targets);
                        switch (loss_fn) {
                            case (NeuralNetwork::loss_function::BCELoss):
                                dL_dout = differentiate_BCELoss(layers[end_idx].layer_output, targets);
                                break;
                            case (NeuralNetwork::loss_function::MSELoss):
                                dL_dout = differentiate_MSELoss(layers[end_idx].layer_output, targets);
                                break;
                            case (NeuralNetwork::loss_function::CrossEntropyLoss):
                                dL_dout = differentiate_CrossEntropyLoss(layers[end_idx].layer_output, class_labels,
                                                                         layers[end_idx].layer_output.cols());
                                break;
                            default:
                                break;
                        }
                    } else {
                        dL_dout = differentiate_MSELoss(layers[layer_idx].layer_output, layers[layer_idx].codes);
                    }
                }

                //Seperate functions so only have to pass the data if necesseray
                //std::optional may be better
                if (first_layer) {
                    calculate_gradients_first_layer(data);
                } else {
                    calculate_gradients(false);
                }
                layers[layer_idx].adam(dL_dW, dL_db);
            }
            first_layer = false;
        }
    }


private:
    std::vector<Layer> layers;
    //Used so don't have to iterate over all layers each time to find layers needed to update weights and codes
    std::vector<std::tuple<int, int>> code_pairs;
    std::vector<std::tuple<int, int, int>> weight_pairs;
    int start_idx;
    int end_idx;


    //Used in gradient calc
    Eigen::MatrixXd dL_dout;
    Eigen::MatrixXd dL_dc;
    Eigen::MatrixXd dL_dW;
    Eigen::VectorXd dL_db;
    Eigen::MatrixXd inputs;

    //Model hyperparams
    size_t n_iter_codes;
    size_t n_iter_weights;
    double lr_codes;
    double lr_weights;
    double mu;
    int criterion;
    double momentum = 0.9;
    loss_function loss_fn;


};


NB_MODULE(fast_altmin, m) {
    m.def("BCELoss", &BCELoss);
    m.def("MSELoss", &MSELoss);
    m.def("differentiate_ReLU", &differentiate_ReLU);
    m.def("differentiate_sigmoid", &differentiate_sigmoid);
    m.def("differentiate_BCELoss", &differentiate_BCELoss);
    m.def("differentiate_MSELoss", &differentiate_MSELoss);
    m.def("hello_world", &hello_world);
    m.def("hello_world_in", &hello_world_in);
    m.def("hello_world_out", &hello_world_out);
    m.def("lin", &lin);
    m.def("ReLU", &ReLU);
    m.def("sigmoid", &sigmoid);
    m.def("ReLU_inplace", &ReLU_inplace);
    m.def("sigmoid_inplace", &sigmoid_inplace);
    m.def("matrix_in", &matrix_in);
    m.def("matrix_out", &matrix_out);
    m.def("matrix_multiplication", &matrix_multiplication);
    m.def("log_softmax", &log_softmax);
    m.def("negative_log_likelihood", &negative_log_likelihood);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("differentiate_CrossEntropyLoss", &differentiate_CrossEntropyLoss);

    nanobind::class_<Layer> Layer(m, "Layer");

    Layer.def(nanobind::init<Layer::layer_type, int, int, int, Eigen::MatrixXd, Eigen::VectorXd, bool, double>())
            .def(nanobind::init<Layer::layer_type, int, int, double>())
            .def("print_info", &Layer::print_info)
            .def("get_codes_for_layer", &Layer::get_codes_for_layer);


    nanobind::enum_<Layer::layer_type>(Layer, "layer_type")
            .value("relu", Layer::layer_type::RELU)
            .value("linear", Layer::layer_type::LINEAR)
            .value("sigmoid", Layer::layer_type::SIGMOID)
            .export_values();


    nanobind::class_<NeuralNetwork> NeuralNetwork(m, "NeuralNetwork");

    NeuralNetwork.def(nanobind::init<NeuralNetwork::loss_function, int, int, int, int, double, double, double>())
            .def("push_back_lin_layer", &NeuralNetwork::push_back_lin_layer)
            .def("push_back_non_lin_layer", &NeuralNetwork::push_back_non_lin_layer)
            .def("construct_pairs", &NeuralNetwork::construct_pairs)
            .def("print_info", &NeuralNetwork::print_info)
            .def("get_codes", &NeuralNetwork::get_codes)
            .def("return_codes", &NeuralNetwork::return_codes)
            .def("return_weights", &NeuralNetwork::return_weights)
            .def("return_bias", &NeuralNetwork::return_bias)
            .def("set_codes", &NeuralNetwork::set_codes)
            .def("set_weights_and_biases", &NeuralNetwork::set_weights_and_biases)
            .def("update_codes", &NeuralNetwork::update_codes)
            .def("update_weights", &NeuralNetwork::update_weights);

    nanobind::enum_<NeuralNetwork::loss_function>(NeuralNetwork, "loss_function")
            .value("BCELoss", NeuralNetwork::loss_function::BCELoss)
            .value("MSELoss", NeuralNetwork::loss_function::MSELoss)
            .value("CrossEntropyLoss", NeuralNetwork::loss_function::CrossEntropyLoss)
            .export_values();

}
