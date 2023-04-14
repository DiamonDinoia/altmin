#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <iostream>
#include <map>
#include <tuple>
#include <vector>

namespace nb = nanobind;

Eigen::MatrixXd lin(const nb::DRef<Eigen::MatrixXd> &input,
                    const nb::DRef<Eigen::MatrixXd> &weight,
                    const nb::DRef<Eigen::MatrixXd> &bias) {
    Eigen::MatrixXd result =
        (input * weight.transpose()).rowwise() + bias.row(0);
    if (result.rows() == 1) { result = ((input * weight.transpose()) + bias); }
    return result;
}

Eigen::MatrixXd ReLU(const nb::DRef<Eigen::MatrixXd> &input) {
    return input.unaryExpr([](double x) { return std::max(x, 0.0); });
}

Eigen::MatrixXd sigmoid(const nb::DRef<Eigen::MatrixXd> &input) {
    return input.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const Eigen::EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// Currently hardcode size of in array
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_codes(
    const std::array<std::string, 7> &model_mods,
    const nb::ndarray<int> &id_codes, const nb::DRef<Eigen::MatrixXd> &inputs,
    std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::VectorXd> &biases,
    Eigen::VectorXd &last_weights) {
    // std::cout << "hi";
    // for (auto const &i : model_mods) { std::cout << i << "\n"; }

    Eigen::MatrixXd result = inputs;

    /*std::cout << weights[0] << "\n\n";
    std::cout << weights[1] << "\n\n";
    std::cout << last_weights << "\n\n";

    std::cout << "Hi";

    std::cout << biases[0] << "\n\n";
    std::cout << biases[1] << "\n\n";
    std::cout << biases[2] << "\n\n";*/

    int x = 0;
    int y = 0;
    for (auto const &i : model_mods) {
        // std::cout << get_shape(result);
        // std::cout << " " << std::to_string(x) << " " << std::to_string(y)
        //<< "\n";
        std::cout << result << "\n"
                  << "\n";
        std::cout << i << "\n";
        std::cout << x << "hi \n";
        if (i.find("Sequential") != std::string::npos) {
            y = x;
        } else if (i.find("Linear") != std::string::npos) {
            if (y > 0) {
                result = lin(result, last_weights, biases[x]);
            } else {
                result = lin(result, weights[x], biases[x]);
            }
        } else if (i.find("ReLU") != std::string::npos) {
            result = ReLU(result);
        } else if (i.find("Sigmoid") != std::string::npos) {
            result = sigmoid(result);
        }
        x += 1;
    }

    return {result, result};
}

NB_MODULE(nanobind_get_codes, m) {
    m.def("get_codes", &get_codes);
    m.doc() = "Forward pass in c++";
}