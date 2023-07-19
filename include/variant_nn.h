//
// Created by mbarbone on 6/26/23.
//

//
// Edited by Tom 6/07/23
// Completly rework neural network class

#ifndef FAST_ALTMIN_NEURAL_NETWORK_H
#define FAST_ALTMIN_NEURAL_NETWORK_H

#include "variant_layers.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>


#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <tuple>
#include <thread>

enum loss_t { BCE, MSE, CrossEntropy };
template <loss_t loss>
class VariantNeuralNetwork{
public:
    VariantNeuralNetwork(){
        const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);
        
    };
    std::vector<std::variant<LinearLayer, LastLinearLayer, ReluLayer, SigmoidLayer>> layers;
    std::vector<std::tuple<int,int>> weight_pairs; 
    bool init_vals = true;
    

    ALTMIN_INLINE void addLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
        const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
            layers.emplace_back(LinearLayer{batch_size, weight, bias, learning_rate});
    }

    ALTMIN_INLINE void addLastLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
        const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
            layers.emplace_back(LastLinearLayer{batch_size, weight, bias, learning_rate});
    }

    ALTMIN_INLINE void addReluLayer(const int n, const int batch_size) noexcept {
        layers.emplace_back(ReluLayer{n, batch_size});
    }

    ALTMIN_INLINE void addSigmoidLayer(const int n, const int batch_size) noexcept {
        layers.emplace_back(SigmoidLayer{n, batch_size});
    }

    ALTMIN_INLINE void construct_pairs() noexcept {
        int prev_idx = 0;
        for (int x = 0; x<layers.size(); x++){
            if (layers[x].index() == 0 || layers[x].index() == 1 ){
                weight_pairs.emplace_back(prev_idx,x);
                prev_idx = x;
            }
        }
    }


    ALTMIN_INLINE Eigen::MatrixXd get_codes(const nanobind::DRef<Eigen::MatrixXd>& inputs, const bool training_mode) noexcept {
        std::visit(CallForward<decltype(inputs)>{inputs}, layers[0]);
        for (int x = 1; x < layers.size();x++){
            Eigen::MatrixXd in = std::visit(get_layer_output, layers[x-1]);
            std::visit(CallForward<decltype(in)>{in}, layers[x]);
        }
        return std::visit(get_layer_output, layers[layers.size()-1]);
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> return_codes() noexcept {
        std::vector<Eigen::MatrixXd> codes; 
        for (auto& layer :layers){
            if (layer.index() == 0){
                codes.emplace_back(std::visit(CallGetCodes{}, layer));
            }
        }
        return codes;
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> return_weights() noexcept {
        std::vector<Eigen::MatrixXd> weights; 
        for (auto& layer:layers){
            if (layer.index() == 0 || layer.index() == 1 ){
                weights.emplace_back(std::visit(CallGetWeights{}, layer));
            }
        }
        return weights;
    }

    ALTMIN_INLINE std::vector<Eigen::VectorXd> return_biases() noexcept {
        std::vector<Eigen::VectorXd> biases;
        for (auto& layer:layers){
            if (layer.index() == 0 || layer.index() == 1 ){
                biases.emplace_back(std::visit(CallGetBiases{}, layer));
            }
        }
        return biases;
    }

//     ALTMIN_INLINE void set_codes(std::vector<nanobind::DRef<Eigen::MatrixXd>> codes)  noexcept {
//         int y = 0;
//         for (auto &layer: layers) {
//             if (layer->layer == layer_type::LINEAR){
//                 layer->set_codes(codes[y]);
//                 y += 1;
//             }
//         }
//     }

//     //https://eigen.tuxfamily.org/dox/TopicPitfalls.html
    ALTMIN_INLINE Eigen::MatrixXd update_codes(const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {
        Eigen::MatrixXd dc;
        if constexpr(loss == loss_t::BCE){ dc = (1/0.003) * differentiate_BCELoss(std::visit(get_layer_output, layers[layers.size()-1]), targets); }
        else if constexpr(loss == loss_t::MSE) {dc =  (1/0.003) * differentiate_MSELoss(std::visit(get_layer_output, layers[layers.size()-1]), targets);}
        else if constexpr(loss == loss_t::CrossEntropy) {
            auto class_labels = Eigen::VectorXd{targets.reshaped()};
            //Try eigen::MatrixXd & at some point
            Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[layers.size()-1]);
            dc = (1/0.003) * 
                differentiate_CrossEntropyLoss(inputs, class_labels, static_cast<int>(inputs.cols()));
        }

        

        for(int x = layers.size()-1; x >= 0; x--){
            //Linear Layer
            if (layers[x].index() == 0){
                std::visit(CallUpdateCodes<decltype(dc)>{dc}, layers[x]);
                // if (x==0){return;}
                dc = differentiate_MSELoss(std::visit(get_layer_output, layers[x]), std::visit(CallGetCodes{}, layers[x]) );
                dc *= std::visit(CallGetWeights{}, layers[x]);
                if ( x==0 ){return dc;}
            //Last Linear Layer
            }else if (layers[x].index() == 1){
                dc *= std::visit(CallGetWeights{}, layers[x]);
            }else{
                Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[x-1]);
                std::visit(CallDifferentiateLayer<decltype(inputs)>{inputs}, layers[x]);
                dc.array() *= std::visit(CallGetDout{}, layers[x]).array();
            }
        }

        return dc;
    }

    ALTMIN_INLINE void update_weights_not_parallel(const nanobind::DRef<Eigen::MatrixXd> &data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd> &targets_nb) noexcept {
        for (int x = 0; x < weight_pairs.size(); x++) {

            auto indexes = weight_pairs[x];
            const int start_idx = std::get<0>(indexes);
            const int end_idx = std::get<1>(indexes);

            //first layer
            if (x == 0) {
                update_weights<true>(data_nb, std::visit(CallGetCodes{}, layers[end_idx]), start_idx, end_idx);
            }
            //last layer
            else if (x==weight_pairs.size()-1){ 
                const int layer_idx = end_idx;
                update_last_weights(std::visit(CallGetCodes{}, layers[start_idx]), targets_nb, start_idx+1, layer_idx);  
            }
            //hidden layers
            else {
                update_weights<false>(std::visit(CallGetCodes{}, layers[start_idx]), std::visit(CallGetCodes{}, layers[end_idx]), start_idx+1, end_idx);
            }
        }
    }


    ALTMIN_INLINE void update_weights_parallel(const nanobind::DRef<Eigen::MatrixXd> &data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd> &targets_nb) noexcept {

# pragma omp parallel for schedule(dynamic) default(none) shared(data_nb, targets_nb, weight_pairs, layers)
        for (int x = 0; x < weight_pairs.size(); x++) {

            auto indexes = weight_pairs[x];
            const int start_idx = std::get<0>(indexes);
            const int end_idx = std::get<1>(indexes);

            //first layer
            if (x == 0) {
                update_weights<true>(data_nb, std::visit(CallGetCodes{}, layers[end_idx]), start_idx, end_idx);
            }
            //last layer
            else if (x==weight_pairs.size()-1){ 
                const int layer_idx = end_idx;
                update_last_weights(std::visit(CallGetCodes{}, layers[start_idx]), targets_nb, start_idx+1, layer_idx);  
            }
            //hidden layers
            else {
                update_weights<false>(std::visit(CallGetCodes{}, layers[start_idx]), std::visit(CallGetCodes{}, layers[end_idx]), start_idx+1, end_idx);
            }
        }
    }


    template <bool first_layer, typename T, typename G>
    ALTMIN_INLINE void update_weights(T& inputs, const G& targets, const int start_idx, const int end_idx) noexcept {
        //Do necessary forward pass first
        //layers[start_idx]->forward(inputs, false);

        std::visit(CallForward<decltype(inputs)>{inputs, false}, layers[start_idx]);


        for (int idx = start_idx+1; idx <= end_idx; idx++){
            Eigen::MatrixXd in = std::visit(get_layer_output, layers[idx-1]);
            std::visit(CallForward<decltype(in)>{in, false}, layers[idx]);
            //layers[idx]->forward(layers[idx-1]->layer_output, false);
        }


        
        Eigen::MatrixXd dout = differentiate_MSELoss(std::visit(get_layer_output, layers[end_idx]), targets );
        Eigen::MatrixXd dW;
        if constexpr (first_layer){
            dW = dout.transpose() * inputs;
        }else{
            dW = dout.transpose() * std::visit(get_layer_output, layers[end_idx-1]);
        }

        Eigen::VectorXd db = dout.colwise().sum();

        std::visit(CallAdam<decltype(dW), decltype(db)>{dW,db}, layers[end_idx]);

        //layers[end_idx]->adam(dW, db);

    }

    template <typename T, typename G>
    ALTMIN_INLINE void update_last_weights(T& inputs,const G& targets, const int start_idx, const int layer_idx) noexcept {
        const int end_idx = layers.size()-1;
        std::visit(CallForward<decltype(inputs)>{inputs, false}, layers[start_idx]);

        for (int idx = start_idx+1; idx <= end_idx; idx++){
            Eigen::MatrixXd in = std::visit(get_layer_output, layers[idx-1]);
            std::visit(CallForward<decltype(in)>{in, false}, layers[idx]);
            //layers[idx]->forward(layers[idx-1]->layer_output, false);
        }

        Eigen::MatrixXd dout;
        if constexpr(loss == loss_t::BCE) { dout = differentiate_BCELoss(std::visit(get_layer_output, layers[end_idx]), targets);}
        else if constexpr(loss == loss_t::MSE) {dout = differentiate_MSELoss(std::visit(get_layer_output, layers[end_idx]), targets);}
        else if constexpr(loss == loss_t::CrossEntropy) {
            auto class_labels = Eigen::VectorXd{targets.reshaped()};
            // dout = differentiate_CrossEntropyLoss(layers[end_idx]->layer_output, class_labels,
            //                                         static_cast<int>(layers[end_idx]->layer_output.cols()));}
            Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[end_idx]);
            dout = (1/0.003) * 
                differentiate_CrossEntropyLoss(inputs, class_labels, static_cast<int>(inputs.cols()));
        }
        
        for (int idx = end_idx; idx > layer_idx; idx--){
            // layers[idx]->differentiate_layer(layers[idx-1]->layer_output);
            // dout.array() *= layers[end_idx]->dout.array();
            Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[idx-1]);
            std::visit(CallDifferentiateLayer<decltype(inputs)>{inputs}, layers[idx]);
            //should be idx but both work
            dout.array() *= std::visit(CallGetDout{}, layers[end_idx]).array();
        }
        
        Eigen::MatrixXd dW = dout.transpose() *  std::visit(get_layer_output, layers[layer_idx-1]);
        Eigen::VectorXd db = dout.colwise().sum();
        std::visit(CallAdam<decltype(dW), decltype(db)>{dW,db}, layers[layer_idx]);
    }


};

template <loss_t loss>
class VariantCNN{
public:
    VariantCNN(){
        const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);
        
    };
    std::vector<std::variant<Conv2dLayer, ReluCNNLayer, MaxPool2dLayer>> layers;
    std::vector<std::tuple<int,int>> weight_pairs; 
    bool init_vals = true;

    //Copy for now
    ALTMIN_INLINE void addConv2dLayer(std::vector<std::vector<Eigen::MatrixXd>> kernels, Eigen::VectorXd bias, int batch_size, int C_in, int height, int width) noexcept{
        layers.emplace_back(Conv2dLayer{kernels, bias, batch_size, C_in, height, width});
    }

    ALTMIN_INLINE void addMaxPool2dLayer(int kernel_size, int stride, int batch_size, int C, int height, int width) noexcept{
        layers.emplace_back(MaxPool2dLayer{kernel_size, stride, batch_size, C, height, width});
    }

   

    ALTMIN_INLINE void addReluCNNLayer(const int N, const int C, const int H, const int W) noexcept {
        layers.emplace_back(ReluCNNLayer{N, C, H, W});
    }


    ALTMIN_INLINE void construct_pairs() noexcept {
        int prev_idx = 0;
        for (int x = 0; x<layers.size(); x++){
            if (layers[x].index() == 0){
                weight_pairs.emplace_back(prev_idx,x);
                prev_idx = x;
            }
        }
    }


    ALTMIN_INLINE std::vector<std::vector<std::vector<Eigen::MatrixXd>>> return_codes_cnn() noexcept {
        std::vector<std::vector<std::vector<Eigen::MatrixXd>>> codes;
        for (auto& layer :layers){
            if (layer.index() == 0){
                codes.emplace_back(std::visit(CallGetCNNCodes{}, layer));
            }
        }
        return codes;
    }

    ALTMIN_INLINE std::vector<std::vector<std::vector<Eigen::MatrixXd>>> return_weights_cnn() noexcept {
        std::vector<std::vector<std::vector<Eigen::MatrixXd>>> weights;
        for (auto& layer :layers){
            if (layer.index() == 0){
                weights.emplace_back(std::visit(CallGetCNNWeights{}, layer));
            }
        }
        return weights;
    }

    ALTMIN_INLINE std::vector<Eigen::VectorXd> return_biases_cnn() noexcept {
        std::vector<Eigen::VectorXd> biases;
        for (auto& layer :layers){
            if (layer.index() == 0){
                biases.emplace_back(std::visit(CallGetCNNBiases{}, layer));
            }
        }
        return biases;
    }


    ALTMIN_INLINE Eigen::MatrixXd get_codes_cnn(const std::vector<std::vector<Eigen::MatrixXd>> &inputs, const bool training_mode) noexcept {
        std::visit(CallForwardCNN<decltype(inputs)>{inputs, training_mode}, layers[0]);
        for (int x = 1; x < layers.size();x++){
            std::vector<std::vector<Eigen::MatrixXd>> in = std::visit(get_layer_output_cnn, layers[x-1]);
            std::visit(CallForwardCNN<decltype(in)>{in, training_mode}, layers[x]);
        }
        //std::vector<std::vector<Eigen::MatrixXd>> res = std::visit(get_layer_output_cnn, layers[0]);
        std::vector<std::vector<Eigen::MatrixXd>> out = std::visit(get_layer_output_cnn, layers[layers.size()-1]);
        return flatten(out);
      
    }

    ALTMIN_INLINE Eigen::MatrixXd get_layer_output_for_ff_update_weights() noexcept{
        std::vector<std::vector<Eigen::MatrixXd>> inputs = std::visit(CallGetCNNCodes{}, layers[3]);
        std::visit(CallForwardCNN<decltype(inputs)>{inputs, false}, layers[4]);
        inputs = std::visit(get_layer_output_cnn, layers[4]);
        std::visit(CallForwardCNN<decltype(inputs)>{inputs, false}, layers[5]);
        inputs = std::visit(get_layer_output_cnn, layers[5]);
        return flatten(inputs);
    }




ALTMIN_INLINE void update_codes_cnn(std::vector<std::vector<Eigen::MatrixXd>> grad) noexcept {
        std::vector<std::vector<Eigen::MatrixXd>> dc = grad;

        

        for(int x = layers.size()-1; x >= 0; x--){
            //Linear Layer
            if (layers[x].index() == 0){
                std::visit(CallCNNUpdateCodes<decltype(dc)>{dc}, layers[x]);
                if (x==0){return;}
                dc = differentiate_MSELoss4d(std::visit(get_layer_output_cnn, layers[x]), std::visit(CallGetCNNCodes{}, layers[x]) );
                std::vector<std::vector<Eigen::MatrixXd>> inputs = std::visit(get_layer_output_cnn, layers[x-1]);
                std::visit(CallCNNDifferentiateLayer<decltype(inputs)>{inputs, dc}, layers[x]);
                dc = std::visit(get_layer_douts_cnn, layers[x]);
            //Last Linear Layer
            }else if (layers[x].index()==2){
                // weird reverse pool thing to get right shape for max pool
                std::vector<std::vector<Eigen::MatrixXd>> inputs = std::visit(get_layer_output_cnn, layers[x-1]);
                std::visit(CallCNNDifferentiateLayer<decltype(inputs)>{inputs, inputs}, layers[x]);
                std::vector<std::vector<Eigen::MatrixXd>> tmp = std::visit(get_layer_douts_cnn, layers[x]);


               
                int stride = std::visit(CallGetStride{}, layers[x]);
                


                for (int n = 0; n < tmp.size(); n++){
                    for(int c=0; c < tmp[0].size(); c++){
                        int x = 0;
                        int y = 0;
                        int count_i = 0;
                        for (int i = 0; i < dc[0][0].rows(); i++){
                            int count_j = 0; 
                            for(int j=0; j <  dc[0][0].cols(); j++){
                                Eigen::Index maxRow, maxCol;
                                tmp[n][c].block(count_i,count_j,stride,stride).maxCoeff(&maxRow, &maxCol);
                                tmp[n][c](count_i+maxRow, count_j+maxCol) = dc[n][c](x,y);
                                y+=1;
                                if (y == dc[0][0].cols()){
                                    y = 0;
                                    x += 1;
                                }
                                count_j += stride;
                            }
                            count_i+= stride;
                        }
                    }
                }

                dc = tmp; 

            }else{
                std::vector<std::vector<Eigen::MatrixXd>> inputs = std::visit(get_layer_output_cnn, layers[x-1]);
                std::visit(CallCNNDifferentiateLayer<decltype(inputs)>{inputs, inputs}, layers[x]);
                std::vector<std::vector<Eigen::MatrixXd>> tmp = std::visit(get_layer_douts_cnn, layers[x]);
                for (int n=0; n < dc.size(); n++){
                    for (int c=0; c < dc[0].size(); c++){
                        dc[n][c].array() *= tmp[n][c].array();
                    }
                }
            }
        }
    }

    ALTMIN_INLINE void update_weights_parallel_cnn(std::vector<std::vector<Eigen::MatrixXd>> inputs) noexcept {
        for (int x = 0; x < weight_pairs.size(); x++) {

            auto indexes = weight_pairs[x];
            const int start_idx = std::get<0>(indexes);
            const int end_idx = std::get<1>(indexes);
            std::cout << "Start " << start_idx << " " << end_idx << std::endl;
            //first layer
            if (x == 0) {
                update_weights_cnn<true>(inputs, std::visit(CallGetCNNCodes{}, layers[end_idx]), start_idx, end_idx);
            }else {
                inputs = std::visit(CallGetCNNCodes{}, layers[start_idx]);
                update_weights_cnn<false>(inputs, std::visit(CallGetCNNCodes{}, layers[end_idx]), start_idx+1, end_idx);
            }
        }
    }

    template <bool first_layer, typename T, typename G>
    ALTMIN_INLINE void update_weights_cnn(T& inputs, const G& targets, const int start_idx, const int end_idx) noexcept {
        //Do necessary forward pass first
        //layers[start_idx]->forward(inputs, false);

        std::visit(CallForwardCNN<decltype(inputs)>{inputs, false}, layers[start_idx]);


        for (int idx = start_idx+1; idx <= end_idx; idx++){
            std::vector<std::vector<Eigen::MatrixXd>> in = std::visit(get_layer_output_cnn, layers[idx-1]);
            std::visit(CallForwardCNN<decltype(in)>{in, false}, layers[idx]);
            //layers[idx]->forward(layers[idx-1]->layer_output, false);
        }


        
        std::vector<std::vector<Eigen::MatrixXd>> dout = differentiate_MSELoss4d(std::visit(get_layer_output_cnn, layers[end_idx]), targets );
        std::vector<std::vector<Eigen::MatrixXd>> dW;
        Eigen::VectorXd db;
        if constexpr (first_layer){
            std::visit(CallCNNDifferentiateLayer<decltype(inputs)>{inputs, dout}, layers[0]);
        }else{
            inputs = std::visit(get_layer_output_cnn, layers[end_idx-1]);
            std::visit(CallCNNDifferentiateLayer<decltype(inputs)>{inputs, dout}, layers[end_idx]);
        }

      

        std::visit(CallAdamCNN{}, layers[end_idx]);

        //layers[end_idx]->adam(dW, db);

    }

    // template <typename T, typename G>
    // ALTMIN_INLINE void update_last_weights(T& inputs,const G& targets, const int start_idx, const int layer_idx) noexcept {
    //     const int end_idx = layers.size()-1;
    //     std::visit(CallForward<decltype(inputs)>{inputs, false}, layers[start_idx]);

    //     for (int idx = start_idx+1; idx <= end_idx; idx++){
    //         Eigen::MatrixXd in = std::visit(get_layer_output, layers[idx-1]);
    //         std::visit(CallForward<decltype(in)>{in, false}, layers[idx]);
    //         //layers[idx]->forward(layers[idx-1]->layer_output, false);
    //     }

    //     Eigen::MatrixXd dout;
    //     if constexpr(loss == loss_t::BCE) { dout = differentiate_BCELoss(std::visit(get_layer_output, layers[end_idx]), targets);}
    //     else if constexpr(loss == loss_t::MSE) {dout = differentiate_MSELoss(std::visit(get_layer_output, layers[end_idx]), targets);}
    //     else if constexpr(loss == loss_t::CrossEntropy) {
    //         auto class_labels = Eigen::VectorXd{targets.reshaped()};
    //         // dout = differentiate_CrossEntropyLoss(layers[end_idx]->layer_output, class_labels,
    //         //                                         static_cast<int>(layers[end_idx]->layer_output.cols()));}
    //         Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[end_idx]);
    //         dout = (1/0.003) * 
    //             differentiate_CrossEntropyLoss(inputs, class_labels, static_cast<int>(inputs.cols()));
    //     }
        
    //     for (int idx = end_idx; idx > layer_idx; idx--){
    //         // layers[idx]->differentiate_layer(layers[idx-1]->layer_output);
    //         // dout.array() *= layers[end_idx]->dout.array();
    //         Eigen::MatrixXd inputs = std::visit(get_layer_output, layers[idx-1]);
    //         std::visit(CallDifferentiateLayer<decltype(inputs)>{inputs}, layers[idx]);
    //         //should be idx but both work
    //         dout.array() *= std::visit(CallGetDout{}, layers[end_idx]).array();
    //     }
        
    //     Eigen::MatrixXd dW = dout.transpose() *  std::visit(get_layer_output, layers[layer_idx-1]);
    //     Eigen::VectorXd db = dout.colwise().sum();
    //     std::visit(CallAdam<decltype(dW), decltype(db)>{dW,db}, layers[layer_idx]);
    // }


};


template <loss_t loss>
class LeNet{
public:
    VariantNeuralNetwork<loss> feed_forward_nn;
    VariantCNN<loss> cnn;
    LeNet(){
        const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);
        
    };
    
    void AddCNN(VariantCNN<loss> cnn){
        this->cnn = cnn;
    }

    void AddFeedForwardNN(VariantNeuralNetwork<loss> feed_forward_nn){
        this->feed_forward_nn = feed_forward_nn;
    }

    ALTMIN_INLINE Eigen::MatrixXd GetCodesLeNet(std::vector<std::vector<Eigen::MatrixXd>> inputs, const bool training_mode) noexcept {
        Eigen::MatrixXd res = cnn.get_codes_cnn(inputs, training_mode);
        res = feed_forward_nn.get_codes(res, training_mode);
        return res;
    }

    std::vector<std::vector<Eigen::MatrixXd>> ReshapeGrad4d(Eigen::MatrixXd grad, int dim_0, int dim_1, int dim_2) noexcept {

        std::vector<std::vector<Eigen::MatrixXd>> res;
        for (int n =0; n < dim_0; n++){
            std::vector<Eigen::MatrixXd> tmp;
            for (int i=0; i < dim_1; i++){
                tmp.emplace_back(grad.block(n,i*dim_2*dim_2,1,dim_2*dim_2).reshaped<Eigen::RowMajor>(dim_2,dim_2));
            } 
            res.emplace_back(tmp);
        }
        return res;
    }

    ALTMIN_INLINE std::vector<std::vector<std::vector<Eigen::MatrixXd>>> ReturnCodesFromCNNLeNet() noexcept {
       return cnn.return_codes_cnn();
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> ReturnCodesFromFFLeNet() noexcept {
       return feed_forward_nn.return_codes();
    }

    ALTMIN_INLINE std::vector<std::vector<std::vector<Eigen::MatrixXd>>> ReturnWeightsFromCNNLeNet() noexcept {
       return cnn.return_weights_cnn();
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> ReturnWeightsFromFFLeNet() noexcept {
       return feed_forward_nn.return_weights();
    }

    ALTMIN_INLINE std::vector<Eigen::VectorXd> ReturnBiasesFromCNNLeNet() noexcept {
       return cnn.return_biases_cnn();
    }

    ALTMIN_INLINE std::vector<Eigen::VectorXd> ReturnBiasesFromFFLeNet() noexcept {
       return feed_forward_nn.return_biases();
    }


    ALTMIN_INLINE void UpdateCodesLeNet(const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {
        Eigen::MatrixXd dc = feed_forward_nn.update_codes(targets);
        //Conver dc to 4d 
        cnn.update_codes_cnn(ReshapeGrad4d(dc, 5, 16, 4));
    }

    ALTMIN_INLINE void UpdateWeightsLeNet(std::vector<std::vector<Eigen::MatrixXd>> data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd> &targets_nb){
        cnn.update_weights_parallel_cnn(data_nb);
        Eigen::MatrixXd inputs = cnn.get_layer_output_for_ff_update_weights();
        feed_forward_nn.update_weights_parallel(inputs,targets_nb);
    }

};

#endif  // FAST_ALTMIN_NEURAL_NETWORK_H
