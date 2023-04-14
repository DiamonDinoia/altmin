#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using namespace Eigen;

MatrixXd sigmoid(const nb::DRef<Eigen::MatrixXd> &input) {
    return input.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

NB_MODULE(nanobind_sigmoid, n) {
    n.def("sigmoid", &sigmoid);
    n.doc() = "sigmoid layer in c++";
}