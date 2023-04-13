#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

namespace nb = nanobind;

using namespace Eigen;

using Eigen::EigenBase;    // <-- Added
using std::ostringstream;  // <-- Added

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// Have to transpose result before sending it back
// I should work out why
MatrixXd lin(const nb::DRef<Eigen::MatrixXd> &input,
             const nb::DRef<Eigen::MatrixXd> &weight,
             const nb::DRef<Eigen::MatrixXd> &bias) {
    MatrixXd result = (input * weight.transpose()).rowwise() + bias.row(0);
    if (result.rows() == 1) { result = ((input * weight.transpose()) + bias); }
    return result;
}

NB_MODULE(nanobind_lin, m) {
    m.def("lin", &lin);
    m.doc() = "linear layer in c++";
}
