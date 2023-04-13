#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using namespace Eigen;

MatrixXd ReLU(const nb::DRef<Eigen::MatrixXd> &input) {
    return input.unaryExpr([](double x) { return std::max(x, 0.0); });
}

NB_MODULE(nanobind_relu, n) {
    n.def("ReLU", &ReLU);
    n.doc() = "ReLU layer in c++";
}