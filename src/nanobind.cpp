#include "nanobind.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <iostream>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>

#define OPTIM_PI 3.14159265358979

int hello_world() {
    std::cout << "c++: Hello World";
    return 0;
}

std::string hello_world_out() { return "Hi python from c++"; }

int         hello_world_in(const std::string &av) {
    std::cout << av;
    return 0;
}

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

double ackley_fn(const Eigen::VectorXd &vals_inp, Eigen::VectorXd *grad_out,
                 void *opt_data) {
    const double x = vals_inp(0);
    const double y = vals_inp(1);

    const double obj_val =
        20 + std::exp(1) -
        20 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + y * y))) -
        std::exp(0.5 *
                 (std::cos(2 * OPTIM_PI * x) + std::cos(2 * OPTIM_PI * y)));

    return obj_val;
}

int main_test() {
    Eigen::VectorXd x =
        2.0 * Eigen::VectorXd::Ones(2);  // initial values: (2,2)

    bool success = optim::de(x, ackley_fn, nullptr);

    if (success) {
        std::cout << "de: Ackley test completed successfully." << std::endl;
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }

    std::cout << "de: solution to Ackley test:\n" << x << std::endl;

    return 0;
}

NB_MODULE(nanobind, m) {
    m.def("BCELoss", &BCELoss);
    m.def("MSELoss", &MSELoss);
    m.def("differentiate_ReLU", &differentiate_ReLU);
    m.def("differentiate_sigmoid", &differentiate_sigmoid);
    m.def("differentiate_last_layer", &differentiate_last_layer);
    m.def("differentiate_mse", &differentiate_mse);
    m.def("hello_world", &hello_world);
    m.def("hello_world_in", &hello_world_in);
    m.def("hello_world_out", &hello_world_out);
    m.def("lin", &lin);
    m.def("ReLU", &ReLU);
    m.def("sigmoid", &sigmoid);
    m.def("ReLU_inplace", &ReLU_inplace);
    m.def("sigmoid_inplace", &sigmoid_inplace);
    m.def("matrix_in", &matrix_in);
    m.def("matrix_out", &matrix_out);
    m.def("matrix_multiplication", &matrix_multiplication);
    m.def("main_test", &main_test);
}
