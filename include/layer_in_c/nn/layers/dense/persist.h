#ifndef LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#define LAYER_IN_C_NN_LAYERS_DENSE_PERSIST_H
#include <highfive/H5File.hpp>
#include <highfive/H5DataSpace.hpp>
#include "layer.h"
#include <layer_in_c/utils/persist.h>
namespace layer_in_c {
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        auto weights = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.weights);
        auto biases = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.biases);
        group.createDataSet("weights", weights);
        group.createDataSet("biases" , biases);
    }
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, HighFive::Group group) {
        save((nn::layers::dense::Layer<DEVICE, SPEC>&)layer, group);
        auto weights = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.output);
        group.createDataSet("output", weights);
    }
    template<typename DEVICE, typename SPEC>
    void save(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackward<DEVICE, SPEC>&)layer, group);
        auto d_weights = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights);
        auto d_biases = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases);
        group.createDataSet("d_weights", d_weights);
        group.createDataSet("d_biases" , d_biases);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(nn::layers::dense::LayerBackwardSGD<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void save(nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        save((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
        auto d_weights_first_order_moment  = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights_first_order_moment);
        auto d_weights_second_order_moment = utils::persist::array_conversion::matrix_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(layer.d_weights_second_order_moment);
        auto d_biases_first_order_moment   = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases_first_order_moment);
        auto d_biases_second_order_moment  = utils::persist::array_conversion::vector_to_std_vector<typename SPEC::T, SPEC::OUTPUT_DIM>(layer.d_biases_second_order_moment);
        group.createDataSet("d_weights_first_order_moment", d_weights_first_order_moment);
        group.createDataSet("d_weights_second_order_moment", d_weights_second_order_moment);
        group.createDataSet("d_biases_first_order_moment" , d_biases_first_order_moment);
        group.createDataSet("d_biases_second_order_moment" , d_biases_second_order_moment);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::Layer<DEVICE, SPEC>& layer, HighFive::Group group) {
        group.getDataSet("weights").read(layer.weights);
        group.getDataSet("biases").read(layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, HighFive::Group group) {
        load((nn::layers::dense::Layer<DEVICE, SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC>
    void load(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackward<DEVICE, SPEC>&)layer, group);
        group.getDataSet("d_weights").read(layer.d_weights);
        group.getDataSet("d_biases").read(layer.d_biases);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(nn::layers::dense::LayerBackwardSGD<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)(layer), group);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    void load(nn::layers::dense::LayerBackwardAdam<DEVICE, SPEC, PARAMETERS>& layer, HighFive::Group group) {
        load((nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>&)layer, group);
        group.getDataSet("d_weights_first_order_moment"). read(layer.d_weights_first_order_moment);
        group.getDataSet( "d_biases_first_order_moment"). read(layer.d_biases_first_order_moment);
        group.getDataSet("d_weights_second_order_moment").read(layer.d_weights_second_order_moment);
        group.getDataSet( "d_biases_second_order_moment").read(layer.d_biases_second_order_moment);
    }
}
#endif
