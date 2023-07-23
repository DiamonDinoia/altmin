#include "variant_nn.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>


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

void test(const nanobind::DRef<Eigen::MatrixXd>& inputs){
    std::cout << inputs << "\n" << std::endl;
    std::cout << inputs.rows() << inputs.cols() << "\n" << std::endl;
    std::cout << "tesr " << inputs.IsRowMajor << std::endl;
}


ALTMIN_INLINE Eigen::MatrixXd py_flatten(std::vector<std::vector<Eigen::MatrixXd>> & inputs){return flatten(inputs);}

ALTMIN_INLINE Eigen::MatrixXd py_lin(const nanobind::DRef<Eigen::MatrixXd> &input,
        const nanobind::DRef<Eigen::MatrixXd> &weight,
        const nanobind::DRef<Eigen::VectorXd> &bias) noexcept {return lin(input,weight,bias);}

ALTMIN_INLINE Eigen::MatrixXd py_ReLU(const nanobind::DRef<Eigen::MatrixXd> &input) noexcept { return ReLU(input); }

ALTMIN_INLINE Eigen::MatrixXd py_sigmoid(const nanobind::DRef<Eigen::MatrixXd> &input) noexcept { return sigmoid(input);}

ALTMIN_INLINE Eigen::MatrixXd py_conv2d(const nanobind::DRef<Eigen::MatrixXd> &input, const nanobind::DRef<Eigen::MatrixXd> &kernel, const double bias,
    const int height, const int width){
        return conv2d(input, kernel, bias, height, width);
}

ALTMIN_INLINE Eigen::MatrixXd py_maxpool2d(const nanobind::DRef<Eigen::MatrixXd> &input,  const int kernel_size, const int stride, const int height, const int width){
    return maxpool2d(input, kernel_size, stride, height, width);
}

ALTMIN_INLINE double py_BCELoss(const nanobind::DRef<Eigen::MatrixXd> &predictions,
                             const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {return BCELoss(predictions, targets);}

ALTMIN_INLINE double py_MSELoss(const nanobind::DRef<Eigen::MatrixXd> &predictions,
                             const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {return MSELoss(predictions, targets);}         

ALTMIN_INLINE void py_log_softmax(nanobind::DRef<Eigen::MatrixXd> input) noexcept {log_softmax(input);}       


ALTMIN_INLINE void py_softmax(nanobind::DRef<Eigen::MatrixXd> input) noexcept {softmax(input);}

ALTMIN_INLINE auto py_one_hot_encoding(const nanobind::DRef<Eigen::VectorXd> &input,
                                               const int num_classes) noexcept {return one_hot_encoding(input, num_classes);}

ALTMIN_INLINE double py_negative_log_likelihood(const nanobind::DRef<Eigen::MatrixXd> &log_likelihoods,
                                             const nanobind::DRef<Eigen::VectorXi> &targets) noexcept {return negative_log_likelihood(log_likelihoods, targets);}

ALTMIN_INLINE double py_cross_entropy_loss(const nanobind::DRef<Eigen::MatrixXd> &input,
                                        const nanobind::DRef<Eigen::VectorXi> &targets) noexcept {return cross_entropy_loss(input, targets);}

ALTMIN_INLINE auto py_differentiate_sigmoid(const nanobind::DRef<Eigen::MatrixXd> &x) noexcept {return differentiate_sigmoid(x);}

ALTMIN_INLINE auto py_differentiate_ReLU(const nanobind::DRef<Eigen::MatrixXd> &x) noexcept {return differentiate_ReLU(x);}

ALTMIN_INLINE auto py_differentiate_maxpool2d(const nanobind::DRef<Eigen::MatrixXd> & inputs, const int kernel_size, const int stride, const int height, const int width) {
    return differentiate_maxpool2d(inputs, kernel_size, stride, height, width);
}

ALTMIN_INLINE auto py_differentiate_conv2d(const nanobind::DRef<Eigen::MatrixXd> & inputs, const nanobind::DRef<Eigen::MatrixXd> & kernels, const int height, const int width, bool flag) {
    return differentiate_conv2d(inputs, kernels,height, width, flag);
}


ALTMIN_INLINE auto py_differentiate_BCELoss(const nanobind::DRef<Eigen::MatrixXd> &output,
                                                    const nanobind::DRef<Eigen::MatrixXd> &target) noexcept {return differentiate_BCELoss(output, target);}

ALTMIN_INLINE auto py_differentiate_MSELoss(const nanobind::DRef<Eigen::MatrixXd> &output,
                                                    const nanobind::DRef<Eigen::MatrixXd> &target) noexcept {return differentiate_MSELoss(output, target);}

ALTMIN_INLINE auto py_differentiate_MSELoss4d(std::vector<std::vector<Eigen::MatrixXd>> output,
                                                    std::vector<std::vector<Eigen::MatrixXd>> target) noexcept {return differentiate_MSELoss4d(output, target);}

ALTMIN_INLINE auto py_differentiate_CrossEntropyLoss(const nanobind::DRef<Eigen::MatrixXd> &output,
                                                             const nanobind::DRef<Eigen::VectorXd> &target,
                                                             const int num_classes) noexcept {return differentiate_CrossEntropyLoss(output, target, num_classes);}

//Credit to Marco Barbone: ///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename V>
constexpr void addVarNNDefs(T& nn) {
    nn.def(nanobind::init<>())
        .def("addSigmoidLayer", &V::addSigmoidLayer)
        .def("addReluLayer", &V::addReluLayer)
        .def("addLinearLayer", &V::addLinearLayer)
        .def("addLastLinearLayer", &V::addLastLinearLayer)
        .def("construct_pairs", &V::construct_pairs)
        .def("get_codes", &V::get_codes)
        .def("update_codes", &V::update_codes)
        .def("update_weights_not_parallel", &V::update_weights_not_parallel)
        .def("update_weights_parallel", &V::update_weights_parallel)
        .def("return_codes", &V::return_codes)
        .def("return_weights", &V::return_weights)
        .def("return_biases", &V::return_biases);
}

template <typename T, typename V>
constexpr void addVarCNNDefs(T& nn) {
    nn.def(nanobind::init<>())
        .def("addConv2dLayer", &V::addConv2dLayer)
        .def("addMaxPool2dLayer", &V::addMaxPool2dLayer)
        .def("addReluCNNLayer", &V::addReluCNNLayer)
        .def("construct_pairs", &V::construct_pairs)
        .def("get_codes_cnn", &V::get_codes_cnn)
        .def("update_codes_cnn", &V::update_codes_cnn)
        .def("return_codes_cnn", &V::return_codes_cnn);
}

template <typename T, typename V>
constexpr void addVarLeNetDefs(T& nn) {
    nn.def(nanobind::init<int, int, int>())
        .def("AddCNN", &V::AddCNN)
        .def("AddFeedForwardNN", &V::AddFeedForwardNN)
        .def("GetCodesLeNet", &V::GetCodesLeNet)
        .def("ReshapeGrad4d", &V::ReshapeGrad4d)
        .def("ReturnCodesFromCNNLeNet", &V::ReturnCodesFromCNNLeNet)
        .def("ReturnCodesFromFFLeNet", &V::ReturnCodesFromFFLeNet)
        .def("ReturnWeightsFromFFLeNet", &V::ReturnWeightsFromFFLeNet)
        .def("ReturnBiasesFromFFLeNet", &V::ReturnBiasesFromFFLeNet)
        .def("ReturnWeightsFromCNNLeNet", &V::ReturnWeightsFromCNNLeNet)
        .def("ReturnBiasesFromCNNLeNet", &V::ReturnBiasesFromCNNLeNet)
        .def("UpdateCodesLeNet", &V::UpdateCodesLeNet)
        .def("UpdateWeightsLeNet", &V::UpdateWeightsLeNet);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


NB_MODULE(fast_altmin, m) {
    m.def("test", &test);
    m.def("BCELoss", &py_BCELoss);
    m.def("MSELoss", &py_MSELoss);
    m.def("differentiate_ReLU", &py_differentiate_ReLU);
    m.def("differentiate_sigmoid", &py_differentiate_sigmoid);
    m.def("differentiate_BCELoss", &py_differentiate_BCELoss);
    m.def("differentiate_MSELoss", &py_differentiate_MSELoss);
    m.def("differentiate_MSELoss4d", &py_differentiate_MSELoss4d);
    m.def("hello_world", &hello_world);
    m.def("hello_world_in", &hello_world_in);
    m.def("hello_world_out", &hello_world_out);
    m.def("lin", &py_lin, nanobind::arg("input").noconvert(), nanobind::arg("weight").noconvert(), nanobind::arg("bias").noconvert());
    m.def("ReLU", &py_ReLU);
    m.def("sigmoid", &py_sigmoid);
    m.def("matrix_in", &matrix_in);
    m.def("matrix_out", &matrix_out);
    m.def("matrix_multiplication", &matrix_multiplication);
    m.def("log_softmax", &py_log_softmax);
    m.def("negative_log_likelihood", &py_negative_log_likelihood);
    m.def("cross_entropy_loss", &py_cross_entropy_loss);
    m.def("differentiate_CrossEntropyLoss", &py_differentiate_CrossEntropyLoss);
    m.def("flatten", &py_flatten);
    m.def("differentiate_maxpool2d", &py_differentiate_maxpool2d);
    m.def("differentiate_conv2d", &py_differentiate_conv2d);
    m.def("conv2d", &py_conv2d);
    m.def("maxpool2d", &py_maxpool2d);

    nanobind::class_<VariantNeuralNetwork<loss_t::BCE>> nnBCE(m, "VariantNeuralNetworkBCE");
    addVarNNDefs<nanobind::class_<VariantNeuralNetwork<loss_t::BCE>>, VariantNeuralNetwork<loss_t::BCE>>(nnBCE);

    nanobind::class_<VariantNeuralNetwork<loss_t::MSE>> nnMSE(m, "VariantNeuralNetworkMSE");
    addVarNNDefs<nanobind::class_<VariantNeuralNetwork<loss_t::MSE>>, VariantNeuralNetwork<loss_t::MSE>>(nnMSE);

    nanobind::class_<VariantNeuralNetwork<loss_t::CrossEntropy>> nnCrossEntropy(m, "VariantNeuralNetworkCrossEntropy");
    addVarNNDefs<nanobind::class_<VariantNeuralNetwork<loss_t::CrossEntropy>>, VariantNeuralNetwork<loss_t::CrossEntropy>>(nnCrossEntropy);

    nanobind::class_<VariantCNN<loss_t::BCE>> cnnBCE(m, "VariantCNNBCE");
    addVarCNNDefs<nanobind::class_<VariantCNN<loss_t::BCE>>, VariantCNN<loss_t::BCE>>(cnnBCE);

    nanobind::class_<VariantCNN<loss_t::CrossEntropy>> cnnCrossEntropy(m, "VariantCNNCrossEntropy");
    addVarCNNDefs<nanobind::class_<VariantCNN<loss_t::CrossEntropy>>, VariantCNN<loss_t::CrossEntropy>>(cnnCrossEntropy);


    nanobind::class_<LeNet<loss_t::CrossEntropy>> LeNetCrossEntropy(m, "LeNetCrossEntropy");
    addVarLeNetDefs<nanobind::class_<LeNet<loss_t::CrossEntropy>>, LeNet<loss_t::CrossEntropy>>(LeNetCrossEntropy);

}

