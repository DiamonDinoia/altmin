#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <iostream>
#include <Eigen/Dense>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

using namespace Eigen;

//using String = std::string;

MatrixXd matrix_multiplication(const nb::DRef<Eigen::MatrixXd> &n, const nb::DRef<Eigen::MatrixXd> &m){
    MatrixXd x = n * m;
    return x;
}

NB_MODULE(nanobind_matrix_multiplication, m){
    m.def("matrix_multiplication", &matrix_multiplication);
    m.doc() = "matrix multiplication in c++";
}

