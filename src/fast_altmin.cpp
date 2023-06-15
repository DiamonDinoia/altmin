#include "fast_altmin.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>

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
    
    //std::cout << "\nstart of apply mods" << std::endl;
    //std::cout << "inputs" << inputs << "\n" << std::endl;
    Eigen::MatrixXd output = inputs;
    
    for (int count = 0; count < end; count++) {
        int i = mods[count];
        if (i == 0) {
            std::cout << "relu" << std::endl;
            ReLU_inplace(output);
            //std::cout << "output after relu" << output << "\n" << std::endl;
        } else if (i == 1) {
            std::cout << "lin" << std::endl;
            output = lin(output, weight, bias);
            //std::cout << "output after lin" << output << "\n" << std::endl;
        } else if (i==2) {
            std::cout << "sigmoid" << std::endl;
            sigmoid_inplace(output);
        }else{
            //std::cout << "cpp: layer not imp" << std::endl;
        }
    }

    //std::cout << "End of apply mods" << std::endl;

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
    size_t n_iter, float lr, bool init_vals, int step, int criterion){
    
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
            //std::cout  << "inputs" << inputs[idx] << "\n" << std::endl;
            output = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], mods[idx].size());

            
            // Need differentiate bceloss to return a cpp type
            // so need to refactor those functions so they have a test function.
            if ((idx + 1) == weights.size() && criterion == 0) {
                dL_doutput = differentiate_BCELoss(output, targets[idx]);
            } else {
                //std::cout << "outputs" << output << "\n" << std::endl;
                //std::cout << "target" << targets[idx] << "\n" << std::endl;
                dL_doutput = differentiate_MSELoss(output, targets[idx]);
            }
            //std::cout << "dl_dout " << dL_doutput << "\n" << std::endl;

            
            //  I think resize will only work for one operation tbh
            for (int x = mods[idx].size() - 1; x > -1; x--) {
                int i = mods[idx][x];
                if (i == 0) {
                    std::cout << "relu" << std::endl;
                    tmp        = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    dL_doutput = dL_doutput.cwiseProduct(differentiate_ReLU(tmp));
                } else if (i == 1) {
                    std::cout << "lin" << std::endl;
                    //std::cout << "Inputs " << inputs[idx] << "\n" << std::endl;
                    tmp   = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    
                    dL_dW = dL_doutput.transpose() * tmp;
                    dL_db = dL_doutput.colwise().sum();
                    break;
                } else if (i == 2) {
                    std::cout << "sigmoid" << std::endl;
                    tmp = apply_mods(weights[idx], biases[idx], mods[idx], inputs[idx], x);
                    dL_doutput =
                        dL_doutput.cwiseProduct(differentiate_sigmoid(tmp));
                } else {
                    std::cout << "Layer not impl yet" << std::endl;
                }
            }
            //std::cout << "dL_dW: " << dL_dW << "\n" << std::endl;
            //std::cout << "dL_db: " << dL_db << "\n" << std::endl;
            
            
            adam_eigen(weight_ms[idx], weight_vs[idx], weights[idx], dL_dW, lr, init_vals,
                    (step + it + 1));
            adam_eigen_bias(bias_ms[idx], bias_vs[idx], biases[idx], dL_db, lr, init_vals,
                            (step + it + 1));
            init_vals = false;
            std::cout << "weight" << weights[idx] << "\n" << std::endl;
            std::cout << "bias " << biases[idx] << "\n" << std::endl;
 
            
        }
        
    }
}

void update_all_weights_CrossEntropyLoss(std::vector<nb::DRef<Eigen::MatrixXd>> weights, std::vector<nb::DRef<Eigen::VectorXd>> biases,
    std::vector<std::vector<int>> mods, std::vector<nb::DRef<Eigen::MatrixXd>> inputs,
    std::vector<nb::DRef<Eigen::MatrixXd>> codes, nb::DRef<Eigen::VectorXi> targets,
    std::vector<nb::DRef<Eigen::MatrixXd>> weight_ms, std::vector<nb::DRef<Eigen::MatrixXd>> weight_vs,
    std::vector<nb::DRef<Eigen::VectorXd>> bias_ms, std::vector<nb::DRef<Eigen::VectorXd>> bias_vs,
    size_t n_iter, float lr, bool init_vals, int step, int criterion){
    
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
                dL_doutput = differentiate_CrossEntropyLoss(output, targets, output.cols());
                //dL_doutput = differentiate_BCELoss(output, targets[idx]);
            } else {
                dL_doutput = differentiate_MSELoss(output, codes[idx]);
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

            std::cout << "dL_dW: " << dL_dW << "\n" << std::endl;
            std::cout << "dL_db: " << dL_db << "\n" << std::endl;
            
            adam_eigen(weight_ms[idx], weight_vs[idx], weights[idx], dL_dW, lr, init_vals,
                    (step + it + 1));
            adam_eigen_bias(bias_ms[idx], bias_vs[idx], biases[idx], dL_db, lr, init_vals,
                            (step + it + 1));
            std::cout << "weight" << weights[idx] << "\n" << std::endl;
            std::cout << "bias " << biases[idx] << "\n" << std::endl;
            init_vals = false;
            return;
        }

    }
}




// Need to prove one iter doesn't make a difference as have lost some
// functionality here It might affect cnns
void update_codes(const nb::DRef<Eigen::MatrixXd> &weight,
                        const nb::DRef<Eigen::VectorXd> &bias,
                        std::vector<int> mods, nb::DRef<Eigen::MatrixXd> codes,
                        const nb::DRef<Eigen::MatrixXd> &targets,
                        size_t is_last_layer, size_t n_iter, double lr, double mu, int criterion) {
    double           momentum = 0.9;

    Eigen::MatrixXd dL_dc;
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd output;
    Eigen::MatrixXd dL_doutput;
    for (size_t it = 0; it < n_iter; it++) {

        output = apply_mods(weight, bias, mods, codes, mods.size());
     
        // Need differentiate bceloss to return a cpp type
        // so need to refactor those functions so they have a test function.
        if (is_last_layer == 1) {
            if (criterion == 0){
                dL_doutput = (1.0 / mu) * differentiate_BCELoss(output, targets);
            }else if(criterion == 2){
                dL_doutput = (1.0 / mu) * differentiate_MSELoss(output, targets);
            }            
        } else {
             dL_doutput = differentiate_MSELoss(output, targets);
        }

        
                    
        //  I think resize will only work for one operation tbh
        for (int x = mods.size() - 1; x > -1; x--) {
            int i = mods[x];
            if (i == 0) {
                //std::cout<<"relu"<<std::endl;
                tmp        = apply_mods(weight, bias, mods, codes, x);
      
               
                dL_dc = dL_doutput.cwiseProduct(differentiate_ReLU(tmp));
            } else if (i == 1) {
                //std::cout<<"lin"<<std::endl;
                dL_doutput = dL_doutput * weight;
                //std::cout << "dL_dout after lin" << dL_doutput << "\n" << std::endl;
            } else if (i == 2) {
                //std::cout<<"sigmoid"<<std::endl;
                //std::cout << "codes before sigmoid " << codes  << "\n" << std::endl;
                tmp = apply_mods(weight, bias, mods, codes, x);
                
                //std::cout << "tmp before sigmoid" << tmp << "\n" << std::endl;
                dL_doutput =
                    dL_doutput.cwiseProduct(differentiate_sigmoid(tmp));
                //std::cout << "dL_dout after sigmoid" << dL_doutput << "\n" << std::endl;
               
            } else {
            }
            
        }
        //std::cout << "dL_dc" << dL_dc << "\n" << std::endl;

        codes = codes - (((1.0 + momentum) * lr) * dL_dc);

        //std::cout << "codes: " << codes << "\n" << std::endl;
    }
}

void update_codes_CrossEntropyLoss(const nb::DRef<Eigen::MatrixXd> &weight,
                        const nb::DRef<Eigen::VectorXd> &bias,
                        std::vector<int> mods, nb::DRef<Eigen::MatrixXd> codes,
                        const nb::DRef<Eigen::VectorXi> &targets,
                        size_t is_last_layer, size_t n_iter, double lr, double mu, int criterion) {
    double           momentum = 0.9;

    Eigen::MatrixXd dL_dc;
    Eigen::MatrixXd tmp;
    Eigen::MatrixXd output;
    Eigen::MatrixXd dL_doutput;
    for (size_t it = 0; it < n_iter; it++) {

        output = apply_mods(weight, bias, mods, codes, mods.size());


        // Need differentiate bceloss to return a cpp type
        // so need to refactor those functions so they have a test function.

        dL_doutput = (1.0 / mu) * differentiate_CrossEntropyLoss(output, targets, output.cols());
            

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

struct Dog {
    std::string name;

    std::string bark() const {
        return name + ": woof!";
    }
};

void test_tuple(std::tuple<int, int> a ){
    std::cout << "hi" << std::endl;
}



class Layer{
    public:
        enum layer_type {
            relu, linear, sigmoid
        };

        Layer(int n, int m, bool has_codes){
            weight = Eigen::MatrixXd(n,m);
            bias = Eigen::VectorXd(m);
            weight_m = Eigen::MatrixXd::Zero(n,m);
            weight_v = Eigen::MatrixXd::Zero(n,m);
            weight_m_t_correct = Eigen::MatrixXd::Zero(n,m);
            weight_v_t_correct = Eigen::MatrixXd::Zero(n,m);
            bias_m = Eigen::VectorXd::Zero(m);
            bias_v = Eigen::VectorXd::Zero(m);
            bias_m_t_correct = Eigen::VectorXd::Zero(m);
            bias_v_t_correct = Eigen::VectorXd::Zero(m);
            //4 is batch size need to pass as param
            codes = Eigen::MatrixXd(4,n);
            layer = linear;
            this->has_codes = has_codes;
            init_vals = true;
            lr = 0.008;
            beta_1 = 0.9;
            beta_2 = 0.999;
            eps    = 1e-08;
            step = 1;
        }
        Layer(layer_type type){
            layer = type;
            this->has_codes = false;
        }

        void initialise_matrices(nb::DRef<Eigen::MatrixXd> weight, nb::DRef<Eigen::VectorXd> bias ){
            this->weight = weight;
            this->bias = bias;
        }

        void print_info(){
            switch(layer){
                case(Layer::layer_type::relu):
                    std::cout << "ReLU" <<  std::endl;
                    break;
                case(Layer::layer_type::linear):
                    std::cout << "Lin: ( " << weight.rows() << " , " << weight.cols() << ") has_codes: " << has_codes  << std::endl; 
                    break;
                case(Layer::layer_type::sigmoid):
                    std::cout << "Sigmoid: " << "\n" << std::endl;
                    break;
            }
        }

        Eigen::MatrixXd get_codes_for_layer(){
            return codes;
        }

        //Move vars declared here out at some point 
        //grad should be a reference 
        void adam(Eigen::MatrixXd grad_weight, Eigen::VectorXd grad_bias){
            if (init_vals == true){
                weight_m = (1 - beta_1) * grad_weight;
                weight_v = (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
              

            }else{
                weight_m = beta_1 * weight_m + (1 - beta_1) * grad_weight;
                weight_v = beta_2 * weight_v + (1 - beta_2) * (grad_weight.cwiseProduct(grad_weight));
            }
            weight_m_t_correct = weight_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
            weight_v_t_correct = weight_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));

            weight = weight - lr * (weight_m_t_correct.cwiseQuotient(
                         (weight_v_t_correct.cwiseSqrt().array() + eps).matrix()));


            if (init_vals == true){
                bias_m = (1 - beta_1) * grad_bias;
                bias_v = (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
                init_vals = false;
            }else{
                bias_m = beta_1 * bias_m + (1 - beta_1) * grad_bias;
                bias_v = beta_2 * bias_v + (1 - beta_2) * (grad_bias.cwiseProduct(grad_bias));
            }
            bias_m_t_correct = bias_m / (1.0 - std::pow(beta_1, static_cast<double>(step)));
            bias_v_t_correct = bias_v / (1.0 - std::pow(beta_2, static_cast<double>(step)));
            bias = bias - lr * (bias_m_t_correct.cwiseQuotient(
                         (bias_v_t_correct.cwiseSqrt().array() + eps).matrix()));
            step = step+1;
        }
        
        friend class NeuralNetwork;
        
    private:
        Eigen::MatrixXd weight;
        Eigen::VectorXd bias;
        Eigen::MatrixXd weight_m; 
        Eigen::MatrixXd weight_v;
        Eigen::VectorXd bias_m;
        Eigen::VectorXd bias_v;
        Eigen::MatrixXd weight_m_t_correct;
        Eigen::MatrixXd weight_v_t_correct;
        Eigen::VectorXd bias_m_t_correct;
        Eigen::VectorXd bias_v_t_correct;

        //Code size is (batch_size, n) so need to pass batch size to initialise the memory for this correctly
        Eigen::MatrixXd codes; 
        layer_type layer;
        bool has_codes;
        bool init_vals;
        float lr;
        float beta_1;
        float beta_2;
        float eps;
        int step;
        
};

class NeuralNetwork{
    public:
        //Setup construtor properly later 
        NeuralNetwork(){
            n_iter = 1;
            lr_codes = 0.3;
            mu = 0.003;
            momentum = 0.9;
            //This will also be an enum in the future
            criterion = 1;
            forward = true;
            init_momentum_vals = true;
        }


        void push_back_layer(Layer layer){
            this->layers.push_back(layer);
        }

        void print_info(){
            int x = 0;
            for (auto i : layers){
                std::cout << "Layer " << x << ": ";
                i.print_info();
                x+=1;
            }
        }

        //Pass whole layer so have access to layer weight and bias
        void apply_layer(Layer &layer){
            switch(layer.layer){
                case(Layer::layer_type::relu):
                    //std::cout << "relu" << std::endl;
                    ReLU_inplace(outputs);
                    break;
                case(Layer::layer_type::linear):
                    //std::cout << "lin" << std::endl;
                    outputs = lin(outputs, layer.weight, layer.bias);
                    if (layer.has_codes == true && forward == true){
                        layer.codes = outputs;
                    }
                    break;
                case(Layer::layer_type::sigmoid):
                    //std::cout << "sigmoid" << std::endl;
                    sigmoid_inplace(outputs);
                    break;
            }
           
        }

        Eigen::MatrixXd get_codes(nb::DRef<Eigen::MatrixXd> inputs){
            outputs = inputs;
            for (int idx = 0 ; idx< layers.size(); idx++){
                apply_layer(layers[idx]);    
            }
            //There will be a better way to do this but for now need to make sure outputs is empty afyer
            //More mem effificent to do at start of every func so once working I might swithc to that
            Eigen::MatrixXd res = outputs;
            outputs = outputs * 0.0; 

            return res;
        }

        std::vector<Eigen::MatrixXd> return_codes(){
            std::vector<Eigen::MatrixXd> codes; 

            for (auto layer : layers){
                if (layer.has_codes == true){
 
     
                    codes.push_back(layer.codes);
                }
            }
            return codes;
        }


        //Need to work out good name
        //This is used in the partial derivatives to calculate the input
        void predict_next_code(Eigen::MatrixXd inputs, int idx, int end_idx){
            outputs = inputs;
            if (idx < end_idx){
                layers[idx].print_info();
                apply_layer(layers[idx]);
            }
           
            if( idx+1 < end_idx){
                layers[idx+1].print_info();
                apply_layer(layers[idx+1]);
            }

            if( idx+2 < end_idx){
                layers[idx+2].print_info();
                apply_layer(layers[idx+2]);
            }      
        }

        //This is used to calculate the derivative of a layer
        void differentiate_layer_for_codes(Eigen::MatrixXd inputs, int start_idx, int end_idx){
            //Layer to calculate the derivative for 
            Layer &layer = layers[end_idx];
           
            //predict_next_code populates outputs 
            //Then used to differentiate the layer
            switch(layer.layer){
                case(Layer::layer_type::relu):
                    predict_next_code(inputs, start_idx, end_idx);
                    dL_dc = dL_dout.cwiseProduct(differentiate_ReLU(outputs));
                    break;
                case(Layer::layer_type::linear):
                    dL_dout = dL_dout * layer.weight;
                    break;
                case(Layer::layer_type::sigmoid):
                    predict_next_code(inputs, start_idx, end_idx);
                    dL_dout = dL_dout.cwiseProduct(differentiate_sigmoid(outputs));
                    break;
            }
        }

        void differentiate_layer_for_weights(Eigen::MatrixXd inputs, int start_idx, int end_idx){
            //Layer to calculate the derivative for 
            Layer &layer = layers[end_idx];
           
            //predict_next_code populates outputs 
            //Then used to differentiate the layer
            switch(layer.layer){
                case(Layer::layer_type::relu):
                    //std::cout << "relu" << std::endl;
                    predict_next_code(inputs, start_idx, end_idx);
                    dL_dout = dL_dout.cwiseProduct(differentiate_ReLU(outputs));
                    break;
                case(Layer::layer_type::linear):
                    //std::cout << "lin" << std::endl;
                    predict_next_code(inputs, start_idx, end_idx);
                    dL_dW = dL_dout.transpose() * outputs;
                    dL_db = dL_dout.colwise().sum();
                    break;
                case(Layer::layer_type::sigmoid):
                    //std::cout << "sigmoid" << std::endl;
                    predict_next_code(inputs, start_idx, end_idx);
                    dL_dout = dL_dout.cwiseProduct(differentiate_sigmoid(outputs));
                    break;
            }
        }

        
        int get_idx_next_layer_with_codes(int idx){
            for(int index = idx+1; index < layers.size(); index++){
                Layer layer = layers[index];
                //layer.print_info();
                if (layer.has_codes){
                    //print_info();

                    return index;
                }
            }
            return layers.size()-1;
        }

        

        void update_codes(nb::DRef<Eigen::MatrixXd> targets){
            forward = false;
            bool last_layer = true;
            int idx_last_code;
            int end_idx;
            Eigen::MatrixXd inputs;

            for (int idx = layers.size()-1; idx > -1 ; idx--){

                
                switch(layers[idx].layer){
                    case(Layer::layer_type::linear):
                        if (layers[idx].has_codes == false){
                            continue;
                        }
                        //Get the id of the next code that had codes or the last layer
                        end_idx = get_idx_next_layer_with_codes(idx);
                        break;
                    default:
                        continue;
                }

                for (size_t it = 0; it<n_iter; it++){

                    //Use the code to predic the next code or model output if next layer
                    inputs = layers[idx].codes;
                    predict_next_code(inputs, idx+1, end_idx+1);
                    
                    if (last_layer == true){
                        dL_dout = (1/mu) * differentiate_BCELoss(outputs, targets);
                    }else{
                        dL_dout = differentiate_MSELoss(outputs,layers[idx_last_code].codes);
                    }
                    last_layer=false;
                    
                    //Copy as relu and sigmoid are in place and don't want to change the codes
                    inputs = layers[idx].codes;
                    outputs = outputs * 0.0;

                    differentiate_layer_for_codes(inputs, idx+1, end_idx);
                    
                    inputs = layers[idx].codes;
                    outputs = outputs * 0.0;
                    if (end_idx -1 != idx){
                        differentiate_layer_for_codes(inputs, idx+1, end_idx-1);
                    }

                    inputs = layers[idx].codes;
                    outputs = outputs * 0.0;
                    if (end_idx -2 != idx){
                        differentiate_layer_for_codes(inputs, idx+1, end_idx-2);
                    }

                    layers[idx].codes = layers[idx].codes - (((1.0 + momentum) * lr_codes) * dL_dc);

                    //
                    idx_last_code = idx;
                    dL_dout = dL_dout * 0.0;
                    dL_dc = dL_dc * 0.0;
                    outputs = outputs * 0.0;
                    //Reset dl_dout to 0
                }

            }
         
        }

        //data gets changed in place and is used in multiple places so would need to be copied anyway so no point passing a reference.
        void update_weights(nb::DRef<Eigen::MatrixXd> data, nb::DRef<Eigen::MatrixXd> targets){
            bool last_layer = false;
            bool first_layer = true;
            int idx_last_code;
            int end_idx;
            Eigen::MatrixXd inputs;
            int start_idx = 0;
            for (int idx = 0; idx < layers.size() ; idx++){

                
                switch(layers[idx].layer){
                    case(Layer::layer_type::linear):
                        //Get the id of the next code that had codes or the last layer
                        end_idx = idx;
                        break;
                    default:
                        continue;
                }

                for (size_t it = 0; it<n_iter; it++){

                    // populate outputs
                    
                    if (first_layer == false){
                        inputs = layers[start_idx].codes;
                    }else{
                        inputs = data;
                        
                    }
                    
                    //std::cout  << "inputs" << inputs << "\n" << std::endl;

                    // if (first_layer = true){
                    //     predict_next_code(inputs, start_idx, idx+1);
                    // }else{
                    //     predict_next_code(inputs, idx+1, end_idx+1);
                    // }

                    predict_next_code(inputs, start_idx, end_idx+1);
                    
                    
                    if (last_layer == true){
                        dL_dout =  differentiate_BCELoss(outputs, targets);
                    }else{
                        //std::cout << "outputs" << outputs << "\n" << std::endl;
                        //std::cout << "target" << layers[idx].codes << "\n" << std::endl;
                        //std::cout << "end_idx" << end_idx << "\n" << std::endl;
                        dL_dout = differentiate_MSELoss(outputs,layers[idx].codes);
                    }

                    //std::cout << "dl_dout " << dL_dout << "\n" << std::endl;
                    
                    
                    if (first_layer == false){
                        inputs = layers[idx].codes;
                    }else{
                        //May have to pass as not a ref
                        inputs = data;
                        
                    }
                    first_layer = false;
                    outputs = outputs * 0.0;

                    //std::cout << "Inputs " << inputs << "\n" << std::endl;
                    differentiate_layer_for_weights(inputs, start_idx, end_idx);

            
                    
                    inputs = layers[idx].codes;
                    outputs = outputs * 0.0;
                    if (end_idx -1 > idx){
                        differentiate_layer_for_weights(inputs, start_idx, end_idx-1);
                    }

                    inputs = layers[idx].codes;
                    outputs = outputs * 0.0;
                    if (end_idx -2 > idx){
                        differentiate_layer_for_weights(inputs, start_idx, end_idx-2);
                    }

                    //std::cout << "dL_dW: " << dL_dW << "\n" << std::endl;
                    //std::cout << "dL_db: " << dL_db << "\n" << std::endl;
                    
                    layers[idx].adam(dL_dW, dL_db);

                   

                    // adam_eigen(weight_ms[idx], weight_vs[idx], weights[idx], dL_dW, lr, init_vals,
                    // (step + it + 1));
                    // adam_eigen_bias(bias_ms[idx], bias_vs[idx], biases[idx], dL_db, lr, init_vals,
                    //         (step + it + 1));
                    start_idx = idx;
                }

            }

        }

        

    private:
        std::vector<Layer> layers; 
        Eigen::MatrixXd outputs; 
        Eigen::MatrixXd dL_dout;
        Eigen::MatrixXd dL_dc;
        Eigen::MatrixXd dL_dW;
        Eigen::VectorXd dL_db;
        Eigen::MatrixXd tmp; 
        size_t n_iter;
        double lr_codes;
        double lr_weights;
        double mu; 
        int criterion;
        double momentum = 0.9; 
        bool forward;
        bool init_momentum_vals;
        
};



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
    m.def("log_softmax", &log_softmax);
    m.def("negative_log_likelihood", &negative_log_likelihood);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("differentiate_CrossEntropyLoss", &differentiate_CrossEntropyLoss);
    m.def("update_codes_CrossEntropyLoss", &update_codes_CrossEntropyLoss);
    m.def("update_all_weights_CrossEntropyLoss", &update_all_weights_CrossEntropyLoss);
    nb::class_<Dog>(m, "Dog")
        .def(nb::init<>())
        .def(nb::init<const std::string &>())
        .def("bark", &Dog::bark)
        .def_rw("name", &Dog::name);
    m.def("test_tuple", &test_tuple);

    nb::class_<Layer> Layer(m, "Layer");
    
    Layer.def(nb::init<int, int, bool>())
        .def(nb::init<Layer::layer_type>())
        .def("initialise_matrices", &Layer::initialise_matrices)
        .def("print_info", &Layer::print_info)
        .def("get_codes_for_layer", &Layer::get_codes_for_layer);

        
    nb::enum_<Layer::layer_type>(Layer, "layer_type")
        .value("relu", Layer::layer_type::relu)
        .value("linear", Layer::layer_type::linear)
        .value("sigmoid", Layer::layer_type::sigmoid)
        .export_values();

    nb::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(nb::init<>())
        .def("push_back_layer", &NeuralNetwork::push_back_layer)
        .def("print_info", &NeuralNetwork::print_info)
        .def("get_codes", &NeuralNetwork::get_codes)
        .def("return_codes", &NeuralNetwork::return_codes)
        .def("update_codes", &NeuralNetwork::update_codes)
        .def("update_weights", &NeuralNetwork::update_weights);
       
}
