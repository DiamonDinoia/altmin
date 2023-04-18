#include "nanobind_criterion.hpp"

NB_MODULE(nanobind_criterion, m) {
    m.def("BCELoss", &BCELoss);
    m.def("MSELoss", &MSELoss);
    m.doc() = "Implementation of criterion in c++";
}
