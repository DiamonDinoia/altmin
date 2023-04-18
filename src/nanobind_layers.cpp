#include "nanobind_layers.hpp"

NB_MODULE(nanobind_layers, m) {
    m.def("lin", &lin);
    m.def("ReLU", &ReLU);
    m.def("sigmoid", &sigmoid);
    m.doc() = "nn layers in c++";
}
