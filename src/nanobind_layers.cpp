#include "nanobind_layers.hpp"

NB_MODULE(nanobind_layers, m) {
    m.def("lin", &lin);
    m.def("ReLU", &ReLU);
    m.def("sigmoid", &sigmoid);
    m.def("ReLU_inplace", &ReLU_inplace);
    m.def("sigmoid_inplace", &sigmoid_inplace);
    m.doc() = "nn layers in c++";
}
