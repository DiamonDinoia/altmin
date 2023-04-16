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

Eigen::MatrixXf lin(const nb::DRef<Eigen::MatrixXf> &input,
                    const nb::DRef<Eigen::MatrixXf> &weight,
                    const nb::DRef<Eigen::MatrixXf> &bias) {
    Eigen::MatrixXf result =
        (input * weight.transpose()).rowwise() + bias.row(0);
    if (result.rows() == 1) { result = ((input * weight.transpose()) + bias); }
    return result;
}

Eigen::MatrixXf ReLU(const nb::DRef<Eigen::MatrixXf> &input) {
    return input.unaryExpr(
        [](float x) { return std::max(x, static_cast<float>(0.0)); });
}

Eigen::MatrixXf sigmoid(const nb::DRef<Eigen::MatrixXf> &input) {
    return input.unaryExpr([](float x) {
        return static_cast<float>(1.0) / (static_cast<float>(1.0) + exp(-x));
    });
}

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const Eigen::EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// Currently hardcode size of in array
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> get_codes(
    const std::vector<std::string> &model_mods,
    const nb::ndarray<int> &id_codes, const nb::DRef<Eigen::MatrixXf> &inputs,
    std::vector<nb::DRef<Eigen::MatrixXf>> tmp) {
    // std::cout << "hi";
    // for (auto const &i : model_mods) { std::cout << i << "\n"; }

    Eigen::MatrixXf result = inputs;

    /*std::cout << weights[0] << "\n\n";
    std::cout << weights[1] << "\n\n";
    std::cout << last_weights << "\n\n";

    std::cout << "Hi";

    std::cout << biases[0] << "\n\n";
    std::cout << biases[1] << "\n\n";
    std::cout << biases[2] << "\n\n";*/

    int                        x   = 0;
    int                        y   = 0;
    nb::DRef<Eigen::MatrixXf> *p_w = tmp.data();

    std::cout << "Input: " << inputs << "\n\n";

    /*for (size_t i = 0; i < tmp.size(); ++i) {
        std::cout << "\n" << *(p_w + i) << "\n";
    }*/

    for (auto const &i : model_mods) {
        // std::cout << get_shape(result);
        // std::cout << " " << std::to_string(x) << " " << std::to_string(y)
        //<< "\n";

        if (i.find("Linear") != std::string::npos) {
            result = lin(result, *(p_w + x), *(p_w + x + 1));
            x += 2;
        } else if (i.find("ReLU") != std::string::npos) {
            result = ReLU(result);
        } else if (i.find("Sigmoid") != std::string::npos) {
            result = sigmoid(result);
        }

        std::cout << i << "\n";
        std::cout << "Result: " << result << "\n\n";
    }

    return {result, result};
}

NB_MODULE(nanobind_get_codes, m) {
    m.def("get_codes", &get_codes);
    m.doc() = "Forward pass in c++";
}