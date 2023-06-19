#include "fast_altmin.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
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

class Layer{
    public:
        enum layer_type {
            relu, linear, sigmoid
        };

        Layer(layer_type layer_type, int batch_size, int n, int m, Eigen::MatrixXd weight, Eigen::VectorXd bias , bool  has_codes, double lr){
            this->weight = weight;
            this->bias = bias;
            weight_m = Eigen::MatrixXd::Zero(n,m);
            weight_v = Eigen::MatrixXd::Zero(n,m);
            weight_m_t_correct = Eigen::MatrixXd::Zero(n,m);
            weight_v_t_correct = Eigen::MatrixXd::Zero(n,m);
            bias_m = Eigen::VectorXd::Zero(m);
            bias_v = Eigen::VectorXd::Zero(m);
            bias_m_t_correct = Eigen::VectorXd::Zero(m);
            bias_v_t_correct = Eigen::VectorXd::Zero(m);
            //4 is batch size need to pass as param
            codes = Eigen::MatrixXd(batch_size,n);
            layer_output = Eigen::MatrixXd(batch_size,n);
            dL_dc = Eigen::MatrixXd(n,m);
            layer = linear;
            this->has_codes = has_codes;
            init_vals = true;
            this->lr = lr;
            beta_1 = 0.9;
            beta_2 = 0.999;
            eps    = 1e-08;
            step = 1;
        }
        Layer(layer_type type, int batch_size, int n, double lr){
            layer = type;
            this->has_codes = false;
            this-> lr = lr;
            layer_output = Eigen::MatrixXd(batch_size,n);
            dL_dout = Eigen::MatrixXd(batch_size,n);
        }


        void print_info(){
            switch(layer){
                case(Layer::layer_type::relu):
                    std::cout << "ReLU" <<  std::endl;
                    std::cout << "layer out: ("  << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
                    break;
                case(Layer::layer_type::linear):
                    std::cout << "Lin: ( " << weight.rows() << " , " << weight.cols() << ") has_codes: " << has_codes  << std::endl;
                    std::cout << "layer out: ("  << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
                    break;
                case(Layer::layer_type::sigmoid):
                    std::cout << "Sigmoid: " << std::endl;
                    std::cout << "layer out: ("  << layer_output.rows() << " , " << layer_output.cols() << ")" << std::endl;
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

        void forward(Eigen::Ref<Eigen::MatrixXd> inputs, bool store_codes){
            switch(layer){
                case(Layer::layer_type::relu):
                    //std::cout << "relu" << std::endl;
                    layer_output = inputs;
                    ReLU_inplace(layer_output);
                    break;
                    
                case(Layer::layer_type::linear):
                    layer_output = lin(inputs, weight, bias);
                    if (has_codes == true && store_codes == true){
                        codes = layer_output;
                    }
                    break;
                    
                case(Layer::layer_type::sigmoid):
                    //std::cout << "sigmoid" << std::endl;
                    layer_output = inputs;
                    sigmoid_inplace(layer_output);
                    break;
                    
            }
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
        Eigen::MatrixXd layer_output;
        Eigen::MatrixXd dL_dout;
        Eigen::MatrixXd dL_dc;
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

        enum loss_function {
            BCELoss, MSELoss, CrossEntropyLoss
        };

        //Setup construtor properly later 
        NeuralNetwork(loss_function loss_fn, int n_iter_weights, int batch_size, int m, double lr_codes){
            n_iter_codes = 1;
            this->n_iter_weights = n_iter_weights;
            this->lr_codes = lr_codes;
            mu = 0.003;
            momentum = 0.9;
            //This will also be an enum in the future
            criterion = 1;
            init_momentum_vals = true;
            this->loss_fn = loss_fn;
            inputs = Eigen::MatrixXd(batch_size,m);
        }


        void push_back_non_lin_layer(Layer::layer_type layer_type, int batch_size, int n, double lr){
            //Have to be shared not unique as python needs access to them as well
            //https://github.com/pybind/pybind11/issues/115
            //maybe nanobind has fixed this by now but I couldn't make it work
            std::shared_ptr<Layer> layer_ptr = std::make_shared<Layer>(Layer(layer_type, batch_size, n, lr));
            layers.push_back(std::move(layer_ptr));
        }

        void push_back_lin_layer(Layer::layer_type layer_type, int batch_size, int n, int m, nb::DRef<Eigen::MatrixXd> weight,
            nb::DRef<Eigen::VectorXd> bias , bool  has_codes,  double lr){
            std::shared_ptr<Layer> layer_ptr = std::make_shared<Layer>(Layer(layer_type, batch_size, n, m, weight, bias, has_codes, lr));
            layers.push_back(std::move(layer_ptr));
            
        }

        int get_idx_next_layer_with_codes(int idx){
            for(int index = idx+1; index < layers.size(); index++){
                //Layer layer = layers[index];
                if (layers[index]->has_codes){
                    return index;
                }
            }
            return layers.size()-1;
        }

        void construct_pairs(){
            int end_idx; 
            int start_idx = 0;
            for (int idx = 0; idx < layers.size(); idx++){
                switch(layers[idx]->layer){
                    case(Layer::layer_type::linear):
                        if (layers[idx]->has_codes == false){
                            weight_pairs.push_back(std::make_tuple(start_idx, idx, get_idx_next_layer_with_codes(idx)));
                            continue;
                        }

                        end_idx = get_idx_next_layer_with_codes(idx);
                        code_pairs.insert(code_pairs.begin(), std::make_tuple(idx, end_idx));
                        weight_pairs.push_back(std::make_tuple(start_idx, idx, idx));
                        start_idx = idx;
                        break;
                    default:
                        continue;
                }
            }
        }

        void print_info(){
            int x = 0;
            for (int idx = 0; idx < layers.size(); idx++){
                std::cout << "Layer " << x << ": ";
                layers[idx]->print_info();
                x+=1;
            }
           
        }

        //If update_codes=false in evaluation mode and the codes aren't being updated
        Eigen::MatrixXd get_codes(nb::DRef<Eigen::MatrixXd> inputs_nb, bool update_codes){
            inputs = inputs_nb;
            layers[0]->forward(inputs, update_codes);
            for (int idx = 1 ; idx< layers.size(); idx++){
                layers[idx]->forward(layers[idx-1]->layer_output, update_codes);
            }
            return layers[layers.size()-1]->layer_output;
        }

        // Could expose the variables in the bindings instead but then they would need to be public
        std::vector<Eigen::MatrixXd> return_codes(){
            std::vector<Eigen::MatrixXd> codes; 

            for (int idx = 0; idx < layers.size(); idx++){
                if (layers[idx]->has_codes == true){
                    codes.push_back(layers[idx]->codes);
                }
            }
            return codes;
        }

        void set_codes(std::vector<nb::DRef<Eigen::MatrixXd>> codes){
            int y = 0 ;
            for (int idx = 0; idx < layers.size(); idx++){
                if (layers[idx]->has_codes == true){
                    layers[idx]->codes = codes[y];
                    y+=1;
                }
               
            }
        }

        std::vector<Eigen::MatrixXd> return_weights(){
            std::vector<Eigen::MatrixXd> weights_vec; 

            for (int idx = 0; idx < layers.size(); idx++){
                switch(layers[idx]->layer){
                    case(Layer::layer_type::linear):
                        weights_vec.push_back(layers[idx]->weight);
                        break;
                    default:
                        continue;
                }
            }
            return weights_vec;
        }

        std::vector<Eigen::VectorXd> return_bias(){
            std::vector<Eigen::VectorXd> bias_vec; 

            for (int idx = 0; idx < layers.size(); idx++){
                switch(layers[idx]->layer){
                    case(Layer::layer_type::linear):
                        bias_vec.push_back(layers[idx]->bias);
                        break;
                    default:
                        continue;
                }
            }
            return bias_vec;
        }

        void set_weights_and_biases(std::vector<nb::DRef<Eigen::MatrixXd>> weights, std::vector<nb::DRef<Eigen::VectorXd>> biases){
            int y = 0 ;
            for (int x = 0; x < layers.size(); x++){
                switch(layers[x]->layer){
                    case(Layer::layer_type::linear):
                        layers[x]->weight = weights[y];
                        layers[x]->bias = biases[y];
                        y+=1;
                        break;
                    default:
                        continue;
                }
            }
        }

        // //Need to work out good name
        // //This is used in the partial derivatives to calculate the input
        int calc_matrix_for_derivative(Eigen::Ref<Eigen::MatrixXd> inputs, int idx, int end_idx){            
            if (idx < end_idx){
                layers[idx]->forward(inputs, false); 
            }else{
                return -1;
            }

            if( idx+1 < end_idx){
                layers[idx+1]->forward(layers[idx]->layer_output, false); 
            }else{
                return idx;
            }

            if( idx+2 < end_idx){
                layers[idx+2]->forward(layers[idx+1]->layer_output, false); 
            }else{
                return idx+1;
            }      

            return idx+2;
        }

        void differentiate_layer(Eigen::Ref<Eigen::MatrixXd> inputs, int start_idx, int end_idx, bool code_derivative){
            //Layer &layer = layers[end_idx];
            int layer_idx;
            switch(layers[end_idx]->layer){
                case(Layer::layer_type::relu):
                    layer_idx = calc_matrix_for_derivative(inputs, start_idx, end_idx);
                    if (code_derivative){
                        if (layer_idx == -1){
                            dL_dc = dL_dout.cwiseProduct(differentiate_ReLU(inputs));
                        }else{
                            dL_dc = dL_dout.cwiseProduct(differentiate_ReLU(layers[layer_idx]->layer_output));
                        }
                    }else{
                        if(layer_idx == -1){
                            dL_dout = dL_dout.cwiseProduct(differentiate_ReLU(inputs));
                        }else{
                            dL_dout = dL_dout.cwiseProduct(differentiate_ReLU(layers[layer_idx]->layer_output));
                        }
                    }  
                    break;
                case(Layer::layer_type::linear):
                    //code derivative = false means weight deriative
                    if (code_derivative){
                        layer_idx = calc_matrix_for_derivative(inputs, start_idx, end_idx);
                        dL_dout = dL_dout * layers[end_idx]->weight;
                    }else{
                        layer_idx = calc_matrix_for_derivative(inputs, start_idx, end_idx);
                        if(layer_idx == -1){
                            dL_dW = dL_dout.transpose() * inputs;
                        }else{
                            dL_dW = dL_dout.transpose() * layers[layer_idx]->layer_output;
                        }
                        dL_db = dL_dout.colwise().sum();
                    }
                    break;
                case(Layer::layer_type::sigmoid):
                    layer_idx = calc_matrix_for_derivative(inputs, start_idx, end_idx);
                    dL_dout = dL_dout.cwiseProduct(differentiate_sigmoid(layers[layer_idx]->layer_output));
                    break;
            }
        }

        void calculate_gradients_first_layer(Eigen::MatrixXd data){
            inputs = data;
            differentiate_layer(inputs, start_idx+1, end_idx, false);
           
            inputs = data;
            if (end_idx -1 > start_idx){
                differentiate_layer(inputs, start_idx+1, end_idx-1, false);
            }

            inputs = data;
            if (end_idx -2 > start_idx){
                differentiate_layer(inputs, start_idx+1, end_idx-2, false);
            }

        }


        void calculate_gradients(bool code_derivative){
             //Copy as relu and sigmoid are in place and don't want to change the codes
            inputs = layers[start_idx]->codes;
            differentiate_layer(inputs, start_idx+1, end_idx, code_derivative);
            
            inputs = layers[start_idx]->codes;
            if (end_idx -1 > start_idx){
                differentiate_layer(inputs, start_idx+1, end_idx-1, code_derivative);
            }

            inputs = layers[start_idx]->codes;
            if (end_idx -2 > start_idx){
                differentiate_layer(inputs, start_idx+1, end_idx-2, code_derivative);
            }


        }

        

        void update_codes(nb::DRef<Eigen::MatrixXd> targets){
            bool last_layer = true;
            int idx_last_code;
            
            if (loss_fn == NeuralNetwork::loss_function::CrossEntropyLoss){
                Eigen::VectorXd targets = Eigen::VectorXd {targets.reshaped()};
            }

            
            for (auto indexes : code_pairs){
                start_idx = std::get<0>(indexes);
                end_idx = std::get<1>(indexes);
                
                for (size_t it = 0; it<n_iter_codes; it++){
                    //reset to 0
                    dL_dout = dL_dout * 0.0;
                    dL_dc = dL_dc * 0.0;
                    //Use the code to predic the next code or model output if next layer
                    inputs = layers[start_idx]->codes;
                    calc_matrix_for_derivative(inputs, start_idx+1, end_idx+1);
                    
                    if (last_layer == true){
                        switch(loss_fn){
                            case(NeuralNetwork::loss_function::BCELoss):
                                dL_dout = (1/mu) * differentiate_BCELoss(layers[layers.size()-1]->layer_output, targets);
                                break;
                            case(NeuralNetwork::loss_function::MSELoss):
                                dL_dout = (1/mu) * differentiate_MSELoss(layers[layers.size()-1]->layer_output, targets);
                                break;
                            case(NeuralNetwork::loss_function::CrossEntropyLoss):
                                dL_dout = (1/mu) * differentiate_CrossEntropyLoss(layers[layers.size()-1]->layer_output, targets, layers[layers.size()-1]->layer_output.cols());
                                break;
                            default:
                                std::cout << "Loss not imp yet" << std::endl;
                                break;
                        }
                    }else{
                        dL_dout = differentiate_MSELoss(layers[idx_last_code]->layer_output, layers[idx_last_code]->codes);
                    }
                    last_layer=false;
                    
                   
                    calculate_gradients(true);
                    layers[start_idx]->codes = layers[start_idx]->codes - (((1.0 + momentum) * lr_codes) * dL_dc);

                    idx_last_code = start_idx;

                }

                

            }
         
        }

        // //data gets changed in place and is used in multiple places so would need to be copied anyway so no point passing a reference.
        void update_weights(nb::DRef<Eigen::MatrixXd> data, nb::DRef<Eigen::MatrixXd> targets){
            bool first_layer = true;
            int layer_idx;
            int tmp;
            if (loss_fn == NeuralNetwork::loss_function::CrossEntropyLoss){
                Eigen::VectorXd targets = Eigen::VectorXd {targets.reshaped()};
            }

            for (auto indexes : weight_pairs){
                start_idx = std::get<0>(indexes);
                layer_idx = std::get<1>(indexes);
                end_idx = std::get<2>(indexes);
                for (size_t it = 0; it<n_iter_weights; it++){
                    //reset to 0
                    dL_dout = dL_dout * 0.0;
                    dL_dW = dL_dW * 0.0;
                    dL_db = dL_db * 0.0;


                    // populate outputs
                    if (first_layer == true){
                        inputs = data;
                        calc_matrix_for_derivative(inputs, start_idx, end_idx+1);
                        dL_dout = differentiate_MSELoss(layers[layer_idx]->layer_output,layers[layer_idx]->codes);
                       
                    }else{
                        inputs = layers[start_idx]->codes;
                      
                        calc_matrix_for_derivative(inputs, start_idx+1, end_idx+1);
                        if (end_idx == (layers.size()-1)){
                            //dL_dout =  differentiate_BCELoss(outputs, targets);
                            switch(loss_fn){
                                case(NeuralNetwork::loss_function::BCELoss):
                                    dL_dout = differentiate_BCELoss(layers[layers.size()-1]->layer_output, targets);
                                    break;
                                case(NeuralNetwork::loss_function::MSELoss):
                                    dL_dout = differentiate_MSELoss(layers[layers.size()-1]->layer_output, targets);
                                    break;
                                case(NeuralNetwork::loss_function::CrossEntropyLoss):
                                    dL_dout = differentiate_CrossEntropyLoss(layers[layers.size()-1]->layer_output, targets, layers[layers.size()-1]->layer_output.cols());
                                    break;
                                default:
                                    break;
                            }
                        }else{
                            dL_dout = differentiate_MSELoss(layers[layer_idx]->layer_output,layers[layer_idx]->codes);
                        }
                        
                    }
                    
                    //Seperate functions so only have to pass the data if necesseray 
                    //std::optional may be better
                    if (first_layer){
                        calculate_gradients_first_layer(data);
                    }else{
                        calculate_gradients(false);
                    }
                  
                    layers[layer_idx]->adam(dL_dW, dL_db);

                }
                first_layer = false;
                

            }

        }

        

    private:
        std::vector<std::shared_ptr<Layer>> layers; 
        std::vector<std::tuple<int,int>> code_pairs;
        std::vector<std::tuple<int,int,int>> weight_pairs;
        Eigen::MatrixXd dL_dout;
        Eigen::MatrixXd dL_dc;
        Eigen::MatrixXd dL_dW;
        Eigen::VectorXd dL_db;
        Eigen::MatrixXd tmp; 
        Eigen::MatrixXd inputs;
        size_t n_iter_codes;
        size_t n_iter_weights;
        double lr_codes;
        double lr_weights;
        double mu; 
        int criterion;
        double momentum = 0.9; 
        bool init_momentum_vals;
        int start_idx;
        int end_idx;
        loss_function loss_fn;
        
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
    m.def("log_softmax", &log_softmax);
    m.def("negative_log_likelihood", &negative_log_likelihood);
    m.def("cross_entropy_loss", &cross_entropy_loss);
    m.def("differentiate_CrossEntropyLoss", &differentiate_CrossEntropyLoss);

    nb::class_<Layer> Layer(m, "Layer");
    
    Layer.def(nb::init<Layer::layer_type, int, int, int , Eigen::MatrixXd, Eigen::VectorXd , bool, double>()  )
        .def(nb::init<Layer::layer_type, int, int, double>())
        .def("print_info", &Layer::print_info)
        .def("get_codes_for_layer", &Layer::get_codes_for_layer);

        
    nb::enum_<Layer::layer_type>(Layer, "layer_type")
        .value("relu", Layer::layer_type::relu)
        .value("linear", Layer::layer_type::linear)
        .value("sigmoid", Layer::layer_type::sigmoid)
        .export_values();



    nb::class_<NeuralNetwork> NeuralNetwork(m, "NeuralNetwork");

    NeuralNetwork.def(nb::init<NeuralNetwork::loss_function, int, int, int, double>())
        .def("push_back_lin_layer", &NeuralNetwork::push_back_lin_layer)
        .def("push_back_non_lin_layer", &NeuralNetwork::push_back_non_lin_layer)
        .def("construct_pairs", &NeuralNetwork::construct_pairs)
        .def("print_info", &NeuralNetwork::print_info)
        .def("get_codes", &NeuralNetwork::get_codes)
        .def("return_codes", &NeuralNetwork::return_codes)
        .def("return_weights", &NeuralNetwork::return_weights)
        .def("return_bias", &NeuralNetwork::return_bias)
        .def("set_codes", &NeuralNetwork::set_codes)
        .def("set_weights_and_biases", &NeuralNetwork::set_weights_and_biases)
        .def("update_codes", &NeuralNetwork::update_codes)
        .def("update_weights", &NeuralNetwork::update_weights);

    nb::enum_<NeuralNetwork::loss_function>(NeuralNetwork, "loss_function")
        .value("BCELoss", NeuralNetwork::loss_function::BCELoss)
        .value("MSELoss", NeuralNetwork::loss_function::MSELoss)
        .value("CrossEntropyLoss", NeuralNetwork::loss_function::CrossEntropyLoss)
        .export_values();
       
}
