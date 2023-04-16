#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

// using String = std::string;
void matrix_in(const nb::DRef<Eigen::MatrixXd> &x) {
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

Eigen::MatrixXd matrix_multiplication(const nb::DRef<Eigen::MatrixXd> &n,
                                      const nb::DRef<Eigen::MatrixXd> &m) {
    Eigen::MatrixXd x = n * m;
    return x;
}

NB_MODULE(nanobind_matrix_funcs, m) {
    m.def("matrix_in", &matrix_in);
    m.def("matrix_in", &matrix_out);
    m.def("matrix_multiplication", &matrix_multiplication);
    m.doc() = "Testing nanobind imp of eigen in c++";
}
