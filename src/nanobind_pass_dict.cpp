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
int pass_weights(std::vector<nb::DRef<Eigen::MatrixXf>> weights) {
    nb::DRef<Eigen::MatrixXf> *p_w = weights.data();

    std::cout << "\n" << *(p_w + 0) << "\n";
    std::cout << "\n" << *(p_w + 1) << "\n";

    return 0;
}

int pass_biases(std::vector<nb::DRef<Eigen::MatrixXf>> biases) {
    nb::DRef<Eigen::MatrixXf> *p_b = biases.data();
    std::cout << "\n" << *(p_b + 0) << "\n";
    std::cout << "\n" << *(p_b + 1) << "\n";
    return 0;
}

int pass_both(std::vector<nb::DRef<Eigen::MatrixXf>> tmp) {
    nb::DRef<Eigen::MatrixXf> *p_w = tmp.data();
    for (size_t i = 0; i < tmp.size(); ++i) {
        std::cout << "\n" << *(p_w + i) << "\n";
    }

    return 0;
}

NB_MODULE(nanobind_pass_dict, m) {
    m.def("pass_weights", &pass_weights);
    m.def("pass_biases", &pass_biases);
    m.def("pass_both", &pass_both);
    m.doc() = "python dict to map in cpp";
}