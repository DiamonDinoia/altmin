#include "nanobind_derivatives.hpp"

NB_MODULE(nanobind_derivatives, m) {
    m.def("differentiate_ReLU", &differentiate_ReLU);
    m.def("differentiate_sigmoid", &differentiate_sigmoid);
    m.def("differentiate_last_layer", &differentiate_last_layer);
    m.def("differentiate_mse", &differentiate_mse);
    m.doc() = "derivatives in c++";
}
