#include "fast_altmin.hpp"

#include <gsl/gsl_multimin.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

// Hello world functions used to test basic input and output for nanobind
int hello_world() {
    std::cout << "c++: Hello World";
    return 0;
}

std::string hello_world_out() { return "Hi python from c++"; }

int         hello_world_in(const std::string &av) {
    std::cout << av;
    return 0;
}

// Matrix functions used to test eigen and nanobind
// At the moment the code uses torch instead of eigen but I'm keeping these for
// now as I would be good to transition back to eigen
void matrix_in(const nanobind::DRef<Eigen::MatrixXd> &x) {
    std::cout << x << std::endl;
}

// Matrix functions used to test eigen and nanobind
// At the moment the code uses torch instead of eigen but I'm keeping these for
// now as I would be good to transition back to eigen
void torch_in(nanobind::ndarray<nanobind::pytorch, float,
                                nanobind::shape<nanobind::any, nanobind::any>>
                  x) {
    torch::Tensor input_tensor =
        torch::from_blob(x.data(), {x.shape(0), x.shape(1)})
            .requires_grad_(true);
    std::cout << input_tensor << std::endl;
}

Eigen::MatrixXd matrix_out() {
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2;
    m(0, 1) = -1;
    m(1, 1) = 4;
    return m;
}

Eigen::MatrixXd matrix_multiplication(
    const nanobind::DRef<Eigen::MatrixXd> &n,
    const nanobind::DRef<Eigen::MatrixXd> &m) {
    Eigen::MatrixXd x = n * m;
    return x;
}

// Do linear layer using tensors
torch::Tensor tensor_lin(torch::Tensor input, torch::Tensor weight,
                         torch::Tensor bias) {
    torch::Tensor res =
        torch::matmul(input, torch::transpose(weight, 1, 0)) + bias;
    return res;
}

// Function to allow for testing linear layer as all my testing is in python atm
void test_tensor_lin(
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        inputs,
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight,
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<nanobind::any>>
        bias,
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        result) {
    torch::Tensor bias_tensor = torch::from_blob(bias.data(), {bias.shape(0)})
                                    .repeat({inputs.shape(0), 1});

    torch::Tensor input_tensor =
        torch::from_blob(inputs.data(), {inputs.shape(0), inputs.shape(1)})
            .requires_grad_(true);

    torch::Tensor weight_tensor =
        torch::from_blob(weight.data(), {weight.shape(0), weight.shape(1)});

    torch::Tensor res = tensor_lin(input_tensor, weight_tensor, bias_tensor);

    std::vector<float> vector(res.data_ptr<float>(),
                              res.data_ptr<float>() + res.numel());

    size_t             k = 0;

    for (size_t i = 0; i < result.shape(0); ++i) {
        for (size_t j = 0; j < result.shape(1); ++j) {
            result(i, j) = vector[k];
            k++;
        }
    }
}

// Both functions below test autograd
// So one is probably redundant so I'll remove it soon
void autograd_example(
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        in_tensor) {
    nanobind::gil_scoped_release no_gil;
    torch::Tensor                x = torch::from_blob(in_tensor.data(),
                                                      {in_tensor.shape(0), in_tensor.shape(1)})
                          .requires_grad_(true);
    // torch::Tensor x = torch::ones({2, 2}, torch::requires_grad());
    torch::Tensor y = x + 2;
    // std::cout << y.grad_fn()->name() << std::endl;
    torch::Tensor out = y.mean();
    out.backward();

    // This code below is very ineffecient but I'm aiming to get it working
    // first Then i'll optimise
    torch::Tensor      x_grad = x.grad();

    std::vector<float> vector(x_grad.data_ptr<float>(),
                              x_grad.data_ptr<float>() + x_grad.numel());

    size_t             k = 0;

    for (size_t i = 0; i < in_tensor.shape(0); ++i) {
        for (size_t j = 0; j < in_tensor.shape(1); ++j) {
            in_tensor(i, j) -= vector[k];
            k++;
        }
    }
}

void test_autograd(
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        inputs,
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight,
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<nanobind::any>>
        bias,
    nanobind::ndarray<nanobind::pytorch, float,
                      nanobind::shape<nanobind::any, nanobind::any>>
        targets) {
    nanobind::gil_scoped_release no_gil;
    torch::Tensor bias_tensor = torch::from_blob(bias.data(), {bias.shape(0)})
                                    .repeat({inputs.shape(0), 1})
                                    .requires_grad_(true);

    torch::Tensor input_tensor =
        torch::from_blob(inputs.data(), {inputs.shape(0), inputs.shape(1)})
            .requires_grad_(true);

    torch::Tensor target_tensor =
        torch::from_blob(targets.data(), {targets.shape(0), targets.shape(1)})
            .requires_grad_(true);

    torch::Tensor weight_tensor =
        torch::from_blob(weight.data(), {weight.shape(0), weight.shape(1)})
            .requires_grad_(true);

    torch::Tensor x    = torch::nn::ReLU()(input_tensor);
    torch::Tensor y    = tensor_lin(x, weight_tensor, bias_tensor);
    torch::Tensor z    = torch::nn::Sigmoid()(y);

    torch::Tensor loss = torch::binary_cross_entropy(z, target_tensor);
    loss.backward();
}

// Implement the adam algorithm in cpp
// This is used to calculate the weight updates using two momentum terms and the
// gradient which we use torch to calc
std::vector<torch::Tensor> adam(torch::Tensor m_t_minus_1,
                                torch::Tensor v_t_minus_1, torch::Tensor val,
                                torch::Tensor grad, float lr, bool init_vals,
                                int step) {
    float beta_1 = 0.9;
    float beta_2 = 0.999;
    float eps    = 1e-08;

    // m_t and v_t initialised as 0, python deals with this dynamically but need
    // to use a boolean to deal with this in cpp
    torch::Tensor m_t;
    torch::Tensor v_t;
    if (init_vals == true) {
        m_t = (1 - beta_1) * grad;
        v_t = (1 - beta_2) * (grad.pow(2));
    } else {
        m_t = beta_1 * m_t_minus_1 + (1 - beta_1) * grad;
        v_t = beta_2 * v_t_minus_1 + (1 - beta_2) * (grad.pow(2));
    }

    torch::Tensor m_t_correct =
        m_t / (1.0 - std::pow(beta_1, static_cast<double>(step)));
    torch::Tensor v_t_correct =
        v_t / (1.0 - std::pow(beta_2, static_cast<double>(step)));

    torch::Tensor res =
        val - lr * (m_t_correct / (torch::sqrt(v_t_correct) + eps));

    // I think there will be a better way to return the vals so I'll look at
    // this.

    std::vector<torch::Tensor> result_vec;
    result_vec.push_back(m_t);
    result_vec.push_back(v_t);
    result_vec.push_back(res);
    return result_vec;
}

// Used to test the adam implementation in cpp
void test_adam(nanobind::ndarray<nanobind::pytorch, float,
                                 nanobind::shape<nanobind::any, nanobind::any>>
                   m_t_minus_1,
               nanobind::ndarray<nanobind::pytorch, float,
                                 nanobind::shape<nanobind::any, nanobind::any>>
                   v_t_minus_1,
               nanobind::ndarray<nanobind::pytorch, float,
                                 nanobind::shape<nanobind::any, nanobind::any>>
                   val,
               nanobind::ndarray<nanobind::pytorch, float,
                                 nanobind::shape<nanobind::any, nanobind::any>>
                    grad,
               bool init_vals) {
    // From blob exposes the given data as a Tensor without taking ownership

    // the original data
    torch::Tensor m_t_minus_1_tensor = torch::from_blob(
        m_t_minus_1.data(), {m_t_minus_1.shape(0), m_t_minus_1.shape(1)});
    torch::Tensor v_t_minus_1_tensor = torch::from_blob(
        v_t_minus_1.data(), {v_t_minus_1.shape(0), v_t_minus_1.shape(1)});
    torch::Tensor val_tensor =
        torch::from_blob(val.data(), {val.shape(0), val.shape(1)});
    torch::Tensor grad_tensor =
        torch::from_blob(grad.data(), {grad.shape(0), grad.shape(1)});

    std::vector<torch::Tensor> res =
        adam(m_t_minus_1_tensor, v_t_minus_1_tensor, val_tensor, grad_tensor,
             0.008, init_vals, 1);

    m_t_minus_1_tensor = res[0];
    v_t_minus_1_tensor = res[1];
    val_tensor         = res[2];

    // Temp as there will be a much better way to do this but just prooving
    // concept for now
    // This is the main problem area rn as also used in the functions that
    // update layers and weights

    std::vector<float> vector(
        m_t_minus_1_tensor.data_ptr<float>(),
        m_t_minus_1_tensor.data_ptr<float>() + m_t_minus_1_tensor.numel());

    size_t k = 0;

    for (size_t i = 0; i < m_t_minus_1.shape(0); ++i) {
        for (size_t j = 0; j < m_t_minus_1.shape(1); ++j) {
            m_t_minus_1(i, j) = vector[k];
            k++;
        }
    }

    std::vector<float> vector_1(
        v_t_minus_1_tensor.data_ptr<float>(),
        v_t_minus_1_tensor.data_ptr<float>() + v_t_minus_1_tensor.numel());

    k = 0;

    for (size_t i = 0; i < v_t_minus_1.shape(0); ++i) {
        for (size_t j = 0; j < v_t_minus_1.shape(1); ++j) {
            v_t_minus_1(i, j) = vector_1[k];
            k++;
        }
    }

    std::vector<float> vector_2(
        val_tensor.data_ptr<float>(),
        val_tensor.data_ptr<float>() + val_tensor.numel());

    k = 0;

    for (size_t i = 0; i < val.shape(0); ++i) {
        for (size_t j = 0; j < val.shape(1); ++j) {
            val(i, j) = vector_1[k];
            k++;
        }
    }
}

// mods[0] is nmode
// mods[1] is lmod
// if = -1 then do nothing
// This implements the logic for updating the codes in cpp
void update_codes(
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
                     bias,
    std::vector<int> mods,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        codes,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
           targets,
    size_t criterion, size_t n_iter, size_t last_layer, float lr) {
    nanobind::gil_scoped_release no_gil;

    // From blob exposes the given data as a Tensor without taking ownership of
    // the original data
    // At the moment requires are data to be double but can change function to
    // allow for declaring tensor data type based on input parameter Would be
    // nice to put this in its own function as lots of functions do it but atm I
    // can't pass a nanobind array to a second cpp function
    torch::Tensor bias_tensor =
        torch::from_blob(bias.data(), {bias.shape(0)}, torch::kFloat64);

    torch::Tensor code_tensor =
        torch::from_blob(codes.data(), {codes.shape(0), codes.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor weight_tensor =
        torch::from_blob(weight.data(), {weight.shape(0), weight.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor targets_tensor =
        torch::from_blob(targets.data(), {targets.shape(0), targets.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor codes_initial =
        code_tensor.detach().clone().requires_grad_(true);

    // Repeat bias tensor to make it equal to batch size
    // Needs to be a seperate tensor as I pass the 1d bias tensor to adam to be
    // updated Could maybe be cleaner
    torch::Tensor bias_tensor_repeat;
    for (size_t x = 0; x < n_iter; x++) {
        bias_tensor_repeat =
            bias_tensor.repeat({codes.shape(0), 1}).requires_grad_(true);
        torch::Tensor in_tensor =
            code_tensor.detach().clone().requires_grad_(true);

        torch::Tensor output;
        torch::Tensor loss = torch::mse_loss(codes_initial, in_tensor);

        // Calculate the loss
        if (mods[0] == 0) {
            output = torch::nn::ReLU()(in_tensor);
        } else if (mods[0] == 1) {
            output = tensor_lin(in_tensor, weight_tensor, bias_tensor_repeat);
        }

        for (int i = 1; i < mods.size(); i++) {
            int index = mods[i];
            if (index == 0) {
                output = torch::nn::ReLU()(output);
            } else if (index == 1) {
                output = tensor_lin(output, weight_tensor, bias_tensor);
            } else {
                output = torch::nn::Sigmoid()(output);
            }
        }

        float mu = 0.003;
        if (last_layer == 1) {
            if (criterion == 0) {
                loss += (1 / mu) *
                        torch::binary_cross_entropy(output, targets_tensor);
            } else if (criterion == 1) {
                loss += (1 / mu) *
                        torch::cross_entropy_loss(output, targets_tensor);
            }
        } else {
            loss += (1 / mu) * (mu * torch::mse_loss(output, targets_tensor));
        }

        loss.backward();

        float momentum = 0.9;
        float lr       = 0.3;
        {
            torch::NoGradGuard no_grad;
            code_tensor.add_(-in_tensor.grad(), (1.0 + momentum) * lr);
        }
    }
}

// Implents the logic for upating the last layer in cpp
void update_last_layer(
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
                     bias,
    std::vector<int> mods,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        inputs,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        targets,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight_m,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight_v,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
        bias_m,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
           bias_v,
    size_t criterion, size_t n_iter, float lr, bool init_vals, int step) {
    nanobind::gil_scoped_release no_gil;

    torch::Tensor                bias_tensor =
        torch::from_blob(bias.data(), {bias.shape(0)}, torch::kFloat64);

    torch::Tensor input_tensor =
        torch::from_blob(inputs.data(), {inputs.shape(0), inputs.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor weight_tensor =
        torch::from_blob(weight.data(), {weight.shape(0), weight.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor targets_tensor =
        torch::from_blob(targets.data(), {targets.shape(0), targets.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor weight_tensor_m = torch::from_blob(
        weight_m.data(), {weight_m.shape(0), weight_m.shape(1)},
        torch::kFloat64);

    torch::Tensor weight_tensor_v = torch::from_blob(
        weight_v.data(), {weight_v.shape(0), weight_v.shape(1)},
        torch::kFloat64);

    // Need to fix this repeat at some point
    torch::Tensor bias_tensor_m =
        torch::from_blob(bias_m.data(), {bias_m.shape(0)}, torch::kFloat64);

    torch::Tensor bias_tensor_v =
        torch::from_blob(bias_v.data(), {bias_v.shape(0)}, torch::kFloat64);

    torch::Tensor bias_tensor_repeat;
    for (int x = 0; x < n_iter; x++) {
        // std::cout << bias_tensor << std::endl;
        bias_tensor_repeat =
            bias_tensor.repeat({inputs.shape(0), 1}).requires_grad_(true);
        torch::Tensor in_tensor =
            input_tensor.detach().clone().requires_grad_(true);
        for (auto i : mods) {
            if (i == 0) {
                in_tensor = torch::nn::ReLU()(in_tensor);
            } else if (i == 1) {
                in_tensor =
                    tensor_lin(in_tensor, weight_tensor, bias_tensor_repeat);
            } else {
                in_tensor = torch::nn::Sigmoid()(in_tensor);
            }
        }

        torch::Tensor loss;
        if (criterion == 0) {
            loss = torch::binary_cross_entropy(in_tensor, targets_tensor);
        }
        loss.backward();

        std::vector<torch::Tensor> res =
            adam(weight_tensor_m, weight_tensor_v, weight_tensor,
                 weight_tensor.grad(), lr, init_vals, (step + x + 1));

        weight_tensor_m -= (weight_tensor_m - res[0]);
        weight_tensor_v -= (weight_tensor_v - res[1]);

        {
            torch::NoGradGuard no_grad;
            weight_tensor -=
                (weight_tensor - res[2].detach()).requires_grad_(true);
        }

        res =
            adam(bias_tensor_m, bias_tensor_v, bias_tensor,
                 torch::reshape(torch::sum(bias_tensor_repeat.grad(), 0, true),
                                {bias.shape(0)}),
                 lr, init_vals, (step + x + 1));

        bias_tensor_m -= (bias_tensor_m - res[0]);
        bias_tensor_v -= (bias_tensor_v - res[1]);

        {
            torch::NoGradGuard no_grad;
            bias_tensor -= (bias_tensor - res[2].detach());
        }

        init_vals = false;
        targets_tensor.grad().zero_();
        weight_tensor.grad().zero_();
        // Don't need to zero bias as we make the new bias_tensor_repeat every
        // iteration
    }
}

// Implements the logic for updating the hidden weights in cpp
void update_hidden_weights(
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
                     bias,
    std::vector<int> mods,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        codes_in,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        codes_out,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight_m,
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        weight_v,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
        bias_m,
    nanobind::ndarray<nanobind::pytorch, double, nanobind::shape<nanobind::any>>
           bias_v,
    size_t n_iter, float lr, bool init_vals, int step) {
    nanobind::gil_scoped_release no_gil;

    torch::Tensor                bias_tensor =
        torch::from_blob(bias.data(), {bias.shape(0)}, torch::kFloat64);

    torch::Tensor code_in_tensor =
        torch::from_blob(codes_in.data(),
                         {codes_in.shape(0), codes_in.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor weight_tensor =
        torch::from_blob(weight.data(), {weight.shape(0), weight.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor code_out_tensor =
        torch::from_blob(codes_out.data(),
                         {codes_out.shape(0), codes_out.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor weight_tensor_m = torch::from_blob(
        weight_m.data(), {weight_m.shape(0), weight_m.shape(1)},
        torch::kFloat64);

    torch::Tensor weight_tensor_v = torch::from_blob(
        weight_v.data(), {weight_v.shape(0), weight_v.shape(1)},
        torch::kFloat64);

    // Need to fix this repeat at some point
    torch::Tensor bias_tensor_m =
        torch::from_blob(bias_m.data(), {bias_m.shape(0)}, torch::kFloat64);

    torch::Tensor bias_tensor_v =
        torch::from_blob(bias_v.data(), {bias_v.shape(0)}, torch::kFloat64);

    torch::Tensor bias_tensor_repeat;
    for (size_t x = 0; x < n_iter; x++) {
        bias_tensor_repeat =
            bias_tensor.repeat({codes_in.shape(0), 1}).requires_grad_(true);
        torch::Tensor in_tensor =
            code_in_tensor.detach().clone().requires_grad_(true);

        torch::Tensor output;
        // std::cout << mods << std::endl;
        if (mods[0] == 0) {
            output = torch::nn::ReLU()(in_tensor);
        } else if (mods[0] == 1) {
            output = tensor_lin(in_tensor, weight_tensor, bias_tensor_repeat);
        }  // do rest layer

        for (int i = 1; i < mods.size(); i++) {
            int index = mods[i];
            if (index == 0) {
                output = torch::nn::ReLU()(output);
            } else if (index == 1) {
                output = tensor_lin(output, weight_tensor, bias_tensor_repeat);
            } else {
                output = torch::nn::Sigmoid()(output);
            }
        }

        torch::Tensor loss = torch::mse_loss(output, code_out_tensor.detach());

        loss.backward();

        std::vector<torch::Tensor> res =
            adam(weight_tensor_m, weight_tensor_v, weight_tensor,
                 weight_tensor.grad(), lr, init_vals, (step + x + 1));

        weight_tensor_m -= (weight_tensor_m - res[0]);
        weight_tensor_v -= (weight_tensor_v - res[1]);

        {
            torch::NoGradGuard no_grad;
            weight_tensor -=
                (weight_tensor - res[2].detach()).requires_grad_(true);
        }

        // Should be bias_tensor[0] but that breaks nanobind here but no in
        // Need to work out correct way to get first row of tensor. Tomorrow
        // problem update_last_weight

        res =
            adam(bias_tensor_m, bias_tensor_v, bias_tensor,
                 torch::reshape(torch::sum(bias_tensor_repeat.grad(), 0, true),
                                {bias.shape(0)}),
                 lr, init_vals, (step + x + 1));

        bias_tensor_m -= (bias_tensor_m - res[0]);
        bias_tensor_v -= (bias_tensor_v - res[1]);

        {
            torch::NoGradGuard no_grad;
            bias_tensor -= (bias_tensor - res[2].detach());
        }

        init_vals = false;
        weight_tensor.grad().zero_();
    }
}

void test_from_blob(
    nanobind::ndarray<nanobind::pytorch, double,
                      nanobind::shape<nanobind::any, nanobind::any>>
        in_data) {
    torch::Tensor in_tensor =
        torch::from_blob(in_data.data(), {in_data.shape(0), in_data.shape(1)},
                         torch::kFloat64)
            .requires_grad_(true);

    torch::Tensor out_tensor =
        torch::ones({in_data.shape(0), in_data.shape(1)});

    {
        torch::NoGradGuard no_grad;
        // Have to set in_tensor requires grad to false so can do inplace
        // opeartions So changes are made to underlying python data
        in_tensor.requires_grad_(false);
        // in_tensor = out_tensor but do in place
        in_tensor -= in_tensor + out_tensor;
    }

    std::cout << in_tensor << std::endl;
}

NB_MODULE(fast_altmin, m) {
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
    m.def("ReLU_autograd", &ReLU_autograd);
    m.def("sigmoid_autograd", &sigmoid_autograd);
    m.def("autograd_example", &autograd_example);
    m.def("test_tensor_lin", &test_tensor_lin);
    m.def("test_autograd", &test_autograd);
    m.def("update_last_layer", &update_last_layer);
    m.def("update_codes", &update_codes);
    m.def("update_hidden_weights", &update_hidden_weights);
    m.def("test_adam", &test_adam);
    m.def("test_from_blob", &test_from_blob);
    m.def("torch_in", &torch_in);
}
