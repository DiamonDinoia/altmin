#include "fast_altmin.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>

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

Eigen::MatrixXd apply_mods(const nb::DRef<Eigen::MatrixXd> &weight,
                           const nb::DRef<Eigen::VectorXd> &bias,
                           std::vector<int>                 mods,
                           const nb::DRef<Eigen::MatrixXd> &inputs, int end) {
    Eigen::MatrixXd output = inputs;

    for (int count = 0; count < end; count++) {
        int i = mods[count];
        if (i == 0) {
            ReLU_inplace(output);
        } else if (i == 1) {
            output = lin(output, weight, bias);
        } else {
            sigmoid_inplace(output);
        }
    }

    return output;
}

void adam_eigen(nb::DRef<Eigen::MatrixXd> m_t, nb::DRef<Eigen::MatrixXd> v_t,
                nb::DRef<Eigen::MatrixXd>        val,
                const nb::DRef<Eigen::MatrixXd> &grad, float lr, bool init_vals,
                int step) {
    float beta_1 = 0.9;
    float beta_2 = 0.999;
    float eps    = 1e-08;

    if (init_vals == true) {
        m_t = (1 - beta_1) * grad;
        v_t = (1 - beta_2) * (grad.cwiseProduct(grad));
    } else {
        m_t = beta_1 * m_t + (1 - beta_1) * grad;
        v_t = beta_2 * v_t + (1 - beta_2) * (grad.cwiseProduct(grad));
    }

    Eigen::MatrixXd m_t_correct =
        m_t / (1.0 - std::pow(beta_1, static_cast<double>(step)));
    Eigen::MatrixXd v_t_correct =
        v_t / (1.0 - std::pow(beta_2, static_cast<double>(step)));

    val = val - lr * (m_t_correct.cwiseQuotient(
                         (v_t_correct.cwiseSqrt().array() + eps).matrix()));
}

void adam_eigen_bias(nb::DRef<Eigen::VectorXd>        m_t,
                     nb::DRef<Eigen::VectorXd>        v_t,
                     nb::DRef<Eigen::VectorXd>        val,
                     const nb::DRef<Eigen::VectorXd> &grad, float lr,
                     bool init_vals, int step) {
    float beta_1 = 0.9;
    float beta_2 = 0.999;
    float eps    = 1e-08;

    if (init_vals == true) {
        m_t = (1 - beta_1) * grad;
        v_t = (1 - beta_2) * (grad.cwiseProduct(grad));
    } else {
        m_t = beta_1 * m_t + (1 - beta_1) * grad;
        v_t = beta_2 * v_t + (1 - beta_2) * (grad.cwiseProduct(grad));
    }

    Eigen::VectorXd m_t_correct =
        m_t / (1.0 - std::pow(beta_1, static_cast<double>(step)));
    Eigen::VectorXd v_t_correct =
        v_t / (1.0 - std::pow(beta_2, static_cast<double>(step)));

    val = val - lr * (m_t_correct.cwiseQuotient(
                         (v_t_correct.cwiseSqrt().array() + eps).matrix()));
}

void update_weights(
    nb::DRef<Eigen::MatrixXd> weight, nb::DRef<Eigen::VectorXd> bias,
    std::vector<int> mods, const nb::DRef<Eigen::MatrixXd> &inputs,
    const nb::DRef<Eigen::MatrixXd> &targets,
    nb::DRef<Eigen::MatrixXd> weight_m, nb::DRef<Eigen::MatrixXd> weight_v,
    nb::DRef<Eigen::VectorXd> bias_m, nb::DRef<Eigen::VectorXd> bias_v,
    size_t is_last_layer, size_t n_iter, float lr, bool init_vals, int step) {
    // Declare necessary vectors outside of for loop
    Eigen::MatrixXd dL_dW;
    Eigen::VectorXd dL_db;
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd output;
    Eigen::MatrixXd dL_doutput;

    // Todo: Move if statement logic outside of for loop
    for (size_t it = 0; it < n_iter; it++) {
        output = apply_mods(weight, bias, mods, inputs, mods.size());

        
        // Need differentiate bceloss to return a cpp type
        // so need to refactor those functions so they have a test function.
        if (is_last_layer == 0) {
            dL_doutput = differentiate_MSELoss(output, targets);
        } else {
            dL_doutput = differentiate_BCELoss(output, targets);
        }
        

        //  I think resize will only work for one operation tbh
        for (int x = mods.size() - 1; x > -1; x--) {
            int i = mods[x];
            if (i == 0) {
                tmp        = apply_mods(weight, bias, mods, inputs, x);
                dL_doutput = dL_doutput.cwiseProduct(differentiate_ReLU(tmp));
            } else if (i == 1) {
                tmp   = apply_mods(weight, bias, mods, inputs, x);
                dL_dW = dL_doutput.transpose() * tmp;
                dL_db = dL_doutput.colwise().sum();
                break;
            } else if (i == 2) {
                tmp = apply_mods(weight, bias, mods, inputs, x);
                dL_doutput =
                    dL_doutput.cwiseProduct(differentiate_sigmoid(tmp));
            } else {
                std::cout << "Layer not impl yet" << std::endl;
            }
        }
        adam_eigen(weight_m, weight_v, weight, dL_dW, lr, init_vals,
                   (step + it + 1));
        adam_eigen_bias(bias_m, bias_v, bias, dL_db, lr, init_vals,
                        (step + it + 1));
        init_vals = false;
    }
}

void update_all_weights(std::vector<nb::DRef<Eigen::MatrixXd>> weights, std::vector<nb::DRef<Eigen::VectorXd>> biases,
    std::vector<std::vector<int>> mods, std::vector<nb::DRef<Eigen::MatrixXd>> inputs,
    std::vector<nb::DRef<Eigen::MatrixXd>> targets,
    std::vector<nb::DRef<Eigen::MatrixXd>> weight_ms, std::vector<nb::DRef<Eigen::MatrixXd>> weight_vs,
    std::vector<nb::DRef<Eigen::VectorXd>> bias_ms, std::vector<nb::DRef<Eigen::VectorXd>> bias_vs,
    size_t n_iter, float lr, bool init_vals, int step){
    
    Eigen::MatrixXd dL_dW;
    Eigen::VectorXd dL_db;
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd output;
    Eigen::MatrixXd dL_doutput;
    
    bool flag = init_vals;
    for (int idx = 0; idx < weights.size(); idx++){

        if (flag){
            init_vals = true;
        }

        // Todo: Move if statement logic outside of for loop
        for (size_t it = 0; it < n_iter; it++) {
            output = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], mods[idx].size());

            
            // Need differentiate bceloss to return a cpp type
            // so need to refactor those functions so they have a test function.
            if ((idx + 1) == weights.size()) {
                dL_doutput = differentiate_BCELoss(output, targets[idx]);
            } else {
                dL_doutput = differentiate_MSELoss(output, targets[idx]);
            }
            

            //  I think resize will only work for one operation tbh
            for (int x = mods[idx].size() - 1; x > -1; x--) {
                int i = mods[idx][x];
                if (i == 0) {
                    tmp        = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    dL_doutput = dL_doutput.cwiseProduct(differentiate_ReLU(tmp));
                } else if (i == 1) {
                    tmp   = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    dL_dW = dL_doutput.transpose() * tmp;
                    dL_db = dL_doutput.colwise().sum();
                    break;
                } else if (i == 2) {
                    tmp = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    dL_doutput =
                        dL_doutput.cwiseProduct(differentiate_sigmoid(tmp));
                } else {
                    std::cout << "Layer not impl yet" << std::endl;
                }
            }
            adam_eigen(weight_ms[idx], weight_vs[idx], weights[idx], dL_dW, lr, init_vals,
                    (step + it + 1));
            adam_eigen_bias(bias_ms[idx], bias_vs[idx], biases[idx], dL_db, lr, init_vals,
                            (step + it + 1));
            init_vals = false;
        }

    }



}
// Need to prove one iter doesn't make a difference as have lost some
// functionality here It might affect cnns
void update_codes(const nb::DRef<Eigen::MatrixXd> &weight,
                        const nb::DRef<Eigen::VectorXd> &bias,
                        std::vector<int> mods, nb::DRef<Eigen::MatrixXd> codes,
                        const nb::DRef<Eigen::MatrixXd> &targets,
                        size_t is_last_layer, size_t n_iter, double lr, double mu) {
    double           momentum = 0.9;

    Eigen::MatrixXd dL_dc;
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd output;
    Eigen::MatrixXd dL_doutput;
    for (size_t it = 0; it < n_iter; it++) {
        output = apply_mods(weight, bias, mods, codes, mods.size());


        // Need differentiate bceloss to return a cpp type
        // so need to refactor those functions so they have a test function.
        if (is_last_layer == 0) {
            dL_doutput = differentiate_MSELoss(output, targets);
        } else {
            dL_doutput = (1.0 / mu) * differentiate_BCELoss(output, targets);
        }

        //  I think resize will only work for one operation tbh
        for (int x = mods.size() - 1; x > -1; x--) {
            int i = mods[x];
            if (i == 0) {
    
                tmp        = apply_mods(weight, bias, mods, codes, x);
      
                
                dL_dc = dL_doutput.cwiseProduct(differentiate_ReLU(tmp));
            } else if (i == 1) {
                dL_doutput = dL_doutput * weight;
            } else if (i == 2) {
                tmp = apply_mods(weight, bias, mods, codes, x);
                dL_doutput =
                    dL_doutput.cwiseProduct(differentiate_sigmoid(tmp));
            } else {
            }
        }

        codes = codes - (((1.0 + momentum) * lr) * dL_dc);
    }
}

NB_MODULE(fast_altmin, m) {
    m.def("BCELoss", &BCELoss);
    m.def("MSELoss", &MSELoss);
    m.def("differentiate_ReLU", &differentiate_ReLU);
    m.def("differentiate_sigmoid", &differentiate_sigmoid);
    m.def("differentiate_BCELoss", &differentiate_BCELoss);
    m.def("differentiate_MSELoss", &differentiate_MSELoss);
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
    m.def("update_weights", &update_weights);
    m.def("update_all_weights", &update_all_weights);
    m.def("update_codes", &update_codes);
    m.def("apply_mods", &apply_mods);
}
