//
// Created by mbarbone on 6/26/23.
//

//
// Edited by Tom 6/07/23
// Completly rework neural network class

#ifndef FAST_ALTMIN_NEURAL_NETWORK_H
#define FAST_ALTMIN_NEURAL_NETWORK_H

#include "layers.h"
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
class NeuralNetwork{
public:
    NeuralNetwork(){
        const auto n_threads = std::thread::hardware_concurrency();
        Eigen::setNbThreads(n_threads);
        
    };
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<std::tuple<int,int>> weight_pairs; 
    bool init_vals = true;
    

    ALTMIN_INLINE void addLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
        const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
            auto ptr = std::make_shared<LinearLayer>(batch_size, weight, bias, learning_rate);
            layers.emplace_back(std::move(ptr));
    }

    ALTMIN_INLINE void addLastLinearLayer(const int batch_size, const nanobind::DRef<Eigen::MatrixXd>& weight,
        const nanobind::DRef<Eigen::VectorXd>& bias, const double learning_rate) noexcept {
            auto ptr = std::make_shared<LastLinearLayer>(batch_size, weight, bias, learning_rate);
            layers.emplace_back(std::move(ptr));
    }

    ALTMIN_INLINE void addReluLayer(const int n, const int batch_size) noexcept {
        auto ptr = std::make_shared<ReluLayer>(n,batch_size);
        layers.emplace_back(std::move(ptr));
    }

    ALTMIN_INLINE void addSigmoidLayer(const int n, const int batch_size) noexcept {
        auto ptr = std::make_shared<SigmoidLayer>(n,batch_size);
        layers.emplace_back(std::move(ptr));
    }

    ALTMIN_INLINE void construct_pairs() noexcept {
        int prev_idx = 0;
        for (int x = 0; x<layers.size(); x++){
            if (layers[x]->layer == layer_type::LINEAR || layers[x]->layer == layer_type::LAST_LINEAR ){
                weight_pairs.emplace_back(prev_idx,x);
                prev_idx = x;
            }
        }
    }


    ALTMIN_INLINE Eigen::MatrixXd get_codes(const nanobind::DRef<Eigen::MatrixXd>& inputs, const bool training_mode) noexcept {
        //not nice but can;t use auto for forward as virtual and nanobind is different to all the layer output inputs so maube easiers to convert once.
        Eigen::MatrixXd in = inputs;
        layers[0]->forward(in,training_mode);
        for (int x = 1; x < layers.size();x++){
            layers[x]->forward(layers[x-1]->layer_output,training_mode);
        }
        return layers[layers.size()-1]->layer_output;
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> return_codes() noexcept {
        std::vector<Eigen::MatrixXd> codes; 
        for (auto layer:layers){
            if (layer->layer == layer_type::LINEAR){
                codes.emplace_back(layer->get_codes());
            }
        }
        return codes;
    }

    ALTMIN_INLINE std::vector<Eigen::MatrixXd> return_weights() noexcept {
        std::vector<Eigen::MatrixXd> weights; 
        for (auto layer:layers){
            if (layer->layer == layer_type::LINEAR || layer->layer == layer_type::LAST_LINEAR){
                weights.emplace_back(layer->get_weight());
            }
        }
        return weights;
    }

    ALTMIN_INLINE std::vector<Eigen::VectorXd> return_biases() noexcept {
        std::vector<Eigen::VectorXd> biases;
        for (auto layer:layers){
            if (layer->layer == layer_type::LINEAR || layer->layer == layer_type::LAST_LINEAR){
                biases.emplace_back(layer->get_bias());
            }
        }
        return biases;
    }

    ALTMIN_INLINE void set_codes(std::vector<nanobind::DRef<Eigen::MatrixXd>> codes)  noexcept {
        int y = 0;
        for (auto &layer: layers) {
            if (layer->layer == layer_type::LINEAR){
                layer->set_codes(codes[y]);
                y += 1;
            }
        }
    }

    //https://eigen.tuxfamily.org/dox/TopicPitfalls.html
    ALTMIN_INLINE void update_codes(const nanobind::DRef<Eigen::MatrixXd> &targets) noexcept {
        Eigen::MatrixXd dc;
        if constexpr(loss == loss_t::BCE){ dc = (1/0.003) * differentiate_BCELoss(layers[layers.size()-1]->layer_output, targets); }
        else if constexpr(loss == loss_t::MSE) {dc =  (1/0.003) * differentiate_MSELoss(layers[layers.size()-1]->layer_output, targets);}
        else if constexpr(loss == loss_t::CrossEntropy) {
            auto class_labels = Eigen::VectorXd{targets.reshaped()};
            dc = (1/0.003) * 
                differentiate_CrossEntropyLoss(layers[layers.size()-1]->layer_output, class_labels,
                 static_cast<int>(layers[layers.size()-1]->layer_output.cols()));
        }

        for(int x = layers.size()-1; x >= 0; x--){
            if (layers[x]->layer == layer_type::LINEAR){
                layers[x]->update_codes(dc);
                if (x==0){return;}
                dc = differentiate_MSELoss(layers[x]->layer_output, layers[x]->get_codes() );
                dc *= layers[x]->get_weight();
            }else if (layers[x]->layer == layer_type::LAST_LINEAR){
                dc *= layers[x]->get_weight();
            }else{
                layers[x]->differentiate_layer(layers[x-1]->layer_output);
                dc.array() *= layers[x]->dout.array();
            }
        }
    }

    ALTMIN_INLINE void update_weights_not_parallel(const nanobind::DRef<Eigen::MatrixXd> &data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd> &targets_nb) noexcept {
        for (int x = 0; x < weight_pairs.size(); x++) {
            auto indexes = weight_pairs[x];
            const int start_idx = std::get<0>(indexes);
            const int end_idx = std::get<1>(indexes);
            //first layer
            if (x == 0) {
                Eigen::MatrixXd data = data_nb;
                update_weights<true>(data, layers[end_idx]->get_codes(), start_idx, end_idx);
            }
            //last layer
            else if (x==weight_pairs.size()-1){ 
                const int layer_idx = end_idx;
                update_last_weights(layers[start_idx]->get_codes(), targets_nb, start_idx+1, layer_idx);  
            }
            //hidden layers
            else {
                update_weights<false>(layers[start_idx]->get_codes(), layers[end_idx]->get_codes(), start_idx+1, end_idx);
            }
        }
    }

    ALTMIN_INLINE void update_weights_parallel(const nanobind::DRef<Eigen::MatrixXd> &data_nb,
                                      const nanobind::DRef<Eigen::MatrixXd> &targets_nb) noexcept {

        
#pragma omp parallel for schedule(dynamic) default(none) shared(data_nb, targets_nb, weight_pairs, layers)
        

        for (int x = 0; x < weight_pairs.size(); x++) {
            auto indexes = weight_pairs[x];
            const int start_idx = std::get<0>(indexes);
            const int end_idx = std::get<1>(indexes);
            //first layer
            if (x == 0) {
                //can't avoid this as can't template forward
                Eigen::MatrixXd data = data_nb;
                update_weights<true>(data, layers[end_idx]->get_codes(), start_idx, end_idx);
            }
            //last layer
            else if (x==weight_pairs.size()-1){ 
                const int layer_idx = end_idx;
                update_last_weights(layers[start_idx]->get_codes(), targets_nb, start_idx+1, layer_idx);  
            }
            //hidden layers
            else {
                update_weights<false>(layers[start_idx]->get_codes(), layers[end_idx]->get_codes(), start_idx+1, end_idx);
            }
        }
    }

    template <bool first_layer, typename T, typename G>
    ALTMIN_INLINE void update_weights(T& inputs, const G& targets, const int start_idx, const int end_idx) noexcept {
        //Do necessary forward pass first
        layers[start_idx]->forward(inputs, false);

        for (int idx = start_idx+1; idx <= end_idx; idx++){
            layers[idx]->forward(layers[idx-1]->layer_output, false);
        }

        
        
        Eigen::MatrixXd dout = differentiate_MSELoss(layers[end_idx]->layer_output, targets );
        Eigen::MatrixXd dW;
        if constexpr (first_layer){
            dW = dout.transpose() * inputs;
        }else{
            dW = dout.transpose() * layers[end_idx-1]->layer_output;
        }
        Eigen::VectorXd db = dout.colwise().sum();
        layers[end_idx]->adam(dW, db);

    }

    template <typename T, typename G>
    ALTMIN_INLINE void update_last_weights(T& inputs,const G& targets, const int start_idx, const int layer_idx) noexcept {
        const int end_idx = layers.size()-1;
        layers[start_idx]->forward(inputs, false);

        for (int idx = start_idx+1; idx <= end_idx; idx++){
            layers[idx]->forward(layers[idx-1]->layer_output, false);
        }

        Eigen::MatrixXd dout;
        if constexpr(loss == loss_t::BCE) { dout = differentiate_BCELoss(layers[end_idx]->layer_output, targets);}
        else if constexpr(loss == loss_t::MSE) {dout = differentiate_MSELoss(layers[end_idx]->layer_output, targets);}
        else if constexpr(loss == loss_t::CrossEntropy) {
            auto class_labels = Eigen::VectorXd{targets.reshaped()};
            dout = differentiate_CrossEntropyLoss(layers[end_idx]->layer_output, class_labels,
                                                    static_cast<int>(layers[end_idx]->layer_output.cols()));}
        
        for (int idx = end_idx; idx > layer_idx; idx--){
            layers[idx]->differentiate_layer(layers[idx-1]->layer_output);
            dout.array() *= layers[end_idx]->dout.array();
        }
        
        Eigen::MatrixXd dW = dout.transpose() * layers[layer_idx-1]->layer_output;
        Eigen::VectorXd db = dout.colwise().sum();
        layers[layer_idx]->adam(dW, db);
    }


};

#endif  // FAST_ALTMIN_NEURAL_NETWORK_H
