#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <functional>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include "nanobind_layers.hpp"

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

float MSELoss(const std::vector<float> &predictions,
              const std::vector<float> &targets) {
    int   N   = predictions.size();
    float sum = static_cast<float>(0.0);
    for (size_t i = 0; i < N; ++i) {
        sum += (targets[i] - predictions[i]) * (targets[i] - predictions[i]);
    }
    sum *= (1 / static_cast<float>(N));
    return sum;
}

// https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
template <typename Derived>
std::string get_shape(const Eigen::EigenBase<Derived> &x) {
    std::ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

// use int map to avoid strings
// 0 = linear
// 1 = ReLU
// 2 = sigmoid
std::tuple<Eigen::MatrixXf, std::vector<Eigen::MatrixXf>> get_codes(
    const std::vector<int> &model_mods, const int num_codes,
    const nb::DRef<Eigen::MatrixXf>       &inputs,
    std::vector<nb::DRef<Eigen::MatrixXf>> model_dict) {
    Eigen::MatrixXf              result = inputs;
    std::vector<Eigen::MatrixXf> codes;
    int                          codes_index = 0;
    int                          x           = 0;
    nb::DRef<Eigen::MatrixXf>   *p_w         = model_dict.data();

    for (auto const &i : model_mods) {
        if (i == 0) {
            result = lin(result, *(p_w + x), *(p_w + x + 1));
            x += 2;
            if (codes_index < num_codes) {
                codes.push_back(result);
                codes_index += 1;
            }
        } else if (i == 1) {
            result = ReLU(result);
        } else if (i == 2) {
            result = sigmoid(result);
        } else {
            std::cout << "Layer not imp yet";
            break;
        }
    }

    return {result, codes};
}

// move compute code loss into this func
int update_codes(std::vector<nb::DRef<Eigen::MatrixXf>> codes,
                 const std::vector<int>                &model_mods,
                 const std::vector<int>                &id_codes,
                 const std::vector<float>              &targets,
                 std::vector<nb::DRef<Eigen::MatrixXf>> model_dict) {
    // add criterion mu lambda_c n_iter lr as params

    int                        model_counter = (model_dict.size()) - 1;
    nb::DRef<Eigen::MatrixXf> *p_w           = model_dict.data();
    nb::DRef<Eigen::MatrixXf> *p_c           = codes.data();

    for (int i = id_codes.size() - 1; i >= 0; --i) {
        Eigen::MatrixXf output;
        size_t          idx = id_codes[i];
        if (std::find(id_codes.begin(), id_codes.end(), idx + 1) !=
            id_codes.end()) {
            int y = model_mods[idx + 1];
            if (y == 0) {
                output = lin(output, *(p_w + model_counter - 1),
                             *(p_w + model_counter));
                model_counter -= 2;
            }
        } else {
            if (idx + 1 < model_mods.size()) {
                int y = model_mods[idx + 1];
                if (y == 1) {
                    output = ReLU(*(p_c + i));
                } else if (y == 2) {
                    output = sigmoid(*(p_c + i));
                } else if (y == 3) {
                    output = *(p_c + i);
                    for (int j = idx + 2; j < model_mods.size(); ++j) {
                        std::cout << "j: " << j << "\n";
                        int x = model_mods[j];
                        std::cout << "x: " << x << "\n";
                        if (x == 0) {
                            output = lin(output, *(p_w + model_counter - 1),
                                         *(p_w + model_counter));
                            model_counter -= 2;
                        } else if (x == 1) {
                            output = ReLU(output);
                        } else if (x == 2) {
                            output = sigmoid(output);
                        } else {
                            std::cout << "Layer not imp yet";
                            break;
                        }
                    }

                } else {
                    std::cout << "Layer not imp yet";
                    break;
                }
            }
            // model_mods[idx + 1] != 3 current method for handling combined
            // last layer should find a more elegant solution
            if (idx + 2 < model_mods.size() && model_mods[idx + 1] != 3) {
                int y = model_mods[idx + 2];
                if (y == 0) {
                    output = lin(output, *(p_w + model_counter - 1),
                                 *(p_w + model_counter));
                    model_counter -= 2;
                } else {
                    std::cout << "Layer not imp yet";
                    break;
                }
            }
        }

        // worry about loss function later
        /*
        if (i == id_codes.size() - 1) {
            // def loss_fn(x): return criterion(x, targets)
        } else {
            // def loss_fn(x): return mu*F.mse_loss(x, codes[i-1])
        }

        for (size_t num_iter = 0; num_iter < 5; ++num_iter) {
            /* optimizer.zero_grad()
            loss = compute_codes_loss(
                codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)

            //loss = (1/mu)*loss_fn(output) + F.mse_loss(codes, codes_targets)
            //if lambda_c > 0.0:
            //loss += (lambda_c/mu)*codes.abs().mean()
            loss.backward()
            optimizer.step()
        }*/
        std::cout << output;
        break;
    }

    // return codes;
    std::cout << "\n end \n";
    return 0;
}

NB_MODULE(nanobind_get_codes, m) {
    m.def("get_codes", &get_codes);
    m.def("update_codes", &update_codes);
    m.doc() = "Forward pass in c++";
}