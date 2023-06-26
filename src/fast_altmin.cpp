

#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "neural_network.h"

// Hello world functions used to test basic input and output for nanobind
int hello_world() {
    std::cout << "c++: Hello World";
    return 0;
}

std::string hello_world_out() { return "Hi python from c++"; }

int hello_world_in(const std::string& av) {
    std::cout << av;
    return 0;
}

// Matrix functions used to test eigen and nanobind
// At the moment the code uses torch instead of eigen but I'm keeping these for
// now as I would be good to transition back to eigen
void matrix_in(const nanobind::DRef<Eigen::MatrixXd>& x) { std::cout << x << std::endl; }

Eigen::MatrixXd matrix_out() {
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2;
    m(0, 1) = -1;
    m(1, 1) = 4;
    return m;
}

Eigen::MatrixXd matrix_multiplication(const nanobind::DRef<Eigen::MatrixXd>& n,
                                      const nanobind::DRef<Eigen::MatrixXd>& m) {
    Eigen::MatrixXd x = n * m;
    return x;
}

ALTMIN_INLINE Eigen::MatrixXd pyReLU(const nanobind::DRef<Eigen::MatrixXd>& input) noexcept { return ReLU(input); }

ALTMIN_INLINE void pyReLU_inplace(nanobind::DRef<Eigen::MatrixXd> input) noexcept { ReLU_inplace(input); }

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
    m.def("ReLU", &pyReLU);
    m.def("sigmoid", &sigmoid);
    m.def("ReLU_inplace", &pyReLU_inplace);
    m.def("sigmoid_inplace", &sigmoid_inplace);
    m.def("matrix_in", &matrix_in);
    m.def("matrix_out", &matrix_out);
    m.def("matrix_multiplication", &matrix_multiplication);
    m.def("log_softmax", &log_softmax);
    m.def("negative_log_likelihood", &negative_log_likelihood);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("differentiate_CrossEntropyLoss", &differentiate_CrossEntropyLoss);

    nanobind::class_<Layer>(m, "Layer")
        .def(nanobind::init<int, int>())
        // .def_rw("n", &Layer::n)
        .def_rw("layer_output", &Layer::layer_output)
        .def_rw("dout", &Layer::dout);

    nanobind::class_<Linear>(m, "Linear")
        .def(nanobind::init<int, int, int, Eigen::MatrixXd, Eigen::VectorXd, double>())
        .def_rw("weights", &Linear::m_weight)
        .def_rw("bias", &Linear::m_bias)
        .def_rw("m_codes", &Linear::m_codes);

    nanobind::class_<LastLinear>(m, "LastLinear")
        .def(nanobind::init<int, int, int, Eigen::MatrixXd, Eigen::VectorXd, double>())
        .def_rw("weights", &LastLinear::m_weight)
        .def_rw("bias", &LastLinear::m_bias)
        .def_rw("m_codes", &LastLinear::m_codes);

    nanobind::class_<NonLinear>(m, "NonLinear")
        .def(nanobind::init<int, int>())
        // .def_rw("n", &Layer::n)
        .def_rw("layer_output", &Layer::layer_output)
        .def_rw("dout", &Layer::dout);

    nanobind::class_<Relu>(m, "ReLU")
        .def(nanobind::init<int, int>())
        // .def_rw("n", &Layer::n)
        .def_rw("layer_output", &Layer::layer_output)
        .def_rw("dout", &Layer::dout);

    nanobind::class_<Sigmoid>(m, "Sigmoid")
        .def(nanobind::init<int, int>())
        // .def_rw("n", &Layer::n)
        .def_rw("layer_output", &Layer::layer_output)
        .def_rw("dout", &Layer::dout);

    // nanobind::class_<NeuralNetwork> NeuralNetwork(m, "NeuralNetwork");

    // NeuralNetwork.def(nanobind::init<NeuralNetwork::loss_function, int, int, int, int, double, double, double>())
    //     .def("push_back_lin_layer", &NeuralNetwork::push_back_lin_layer)
    //     .def("push_back_non_lin_layer", &NeuralNetwork::push_back_non_lin_layer)
    //     .def("construct_pairs", &NeuralNetwork::construct_pairs)
    //     .def("print_info", &NeuralNetwork::print_info)
    //     .def("get_codes", &NeuralNetwork::get_codes)
    //     .def("return_codes", &NeuralNetwork::return_codes)
    //     .def("return_weights", &NeuralNetwork::return_weights)
    //     .def("return_bias", &NeuralNetwork::return_bias)
    //     .def("set_codes", &NeuralNetwork::set_codes)
    //     .def("set_weights_and_biases", &NeuralNetwork::set_weights_and_biases)
    //     .def("update_codes", &NeuralNetwork::update_codes)
    //     .def("update_weights", &NeuralNetwork::update_weights);

    // nanobind::enum_<NeuralNetwork::loss_function>(NeuralNetwork, "loss_function")
    //     .value("BCELoss", NeuralNetwork::loss_function::BCELoss)
    //     .value("MSELoss", NeuralNetwork::loss_function::MSELoss)
    //     .value("CrossEntropyLoss", NeuralNetwork::loss_function::CrossEntropyLoss)
    //     .export_values();
}
