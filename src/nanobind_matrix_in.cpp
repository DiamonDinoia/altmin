#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace Eigen;

// using String = std::string;

void matrix_in(const nb::DRef<Eigen::MatrixXd> &x) {
    std::cout << x << std::endl;
}

NB_MODULE(nanobind_matrix_in, m) {
    m.def("matrix_in", &matrix_in);
    m.doc() = "return matrix in c++";
}
