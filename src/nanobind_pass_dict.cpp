#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <map>
#include <tuple>
#include <vector>

namespace nb = nanobind;

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const Eigen::EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// Currently hardcode size of in array
// Have to split into two to get working
int pass_dict(std::vector<Eigen::MatrixXd> &weights,
              std::vector<Eigen::VectorXd> &biases) {
    std::cout << weights[0] << "\n";
    // std::cout << weights[1] << "\n";
    std::cout << biases[0] << get_shape(biases[0]) << "\n";
    // std::cout << biases[1] << get_shape(biases[1]) << "\n";
    return 0;
}

NB_MODULE(nanobind_pass_dict, m) {
    m.def("pass_dict", &pass_dict);
    m.doc() = "python dict to map in cpp";
}