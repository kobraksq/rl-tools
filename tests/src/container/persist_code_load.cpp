#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/containers/persist_code.h>
#include <layer_in_c/nn/layers/dense/operations_cpu.h>
#include <layer_in_c/nn_models/mlp/operations_cpu.h>

namespace lic = layer_in_c;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "../../../data/test_layer_in_c_container_persist_matrix.h"

constexpr bool const_declaration = false;

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> orig;
    lic::malloc(device, orig);
    lic::randn(device, orig, rng);
    std::cout << "orig: " << std::endl;
    lic::print(device, orig);
    std::cout << "loaded: " << std::endl;
    lic::print(device, matrix_1::container);

    auto abs_diff = lic::abs_diff(device, orig, matrix_1::container);
    ASSERT_FLOAT_EQ(0, abs_diff);
}

#include "../../../data/test_layer_in_c_nn_layers_dense_persist_code.h"

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::nn::layers::dense::Layer<lic::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, lic::nn::activation_functions::ActivationFunction::RELU>> layer;
    lic::malloc(device, layer);
    lic::init_kaiming(device, layer, rng);
    lic::increment(layer.weights.parameters, 2, 1, 10);
    auto abs_diff = lic::abs_diff(device, layer, layer_1::layer);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST_DENSE_LAYER_ADAM){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    using OPTIMIZER_PARAMETERS = lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::nn::layers::dense::LayerBackwardGradient<lic::nn::layers::dense::Specification<DTYPE, typename DEVICE::index_t, 3, 3, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::parameters::Adam>> layer;
    lic::malloc(device, layer);
    lic::init_kaiming(device, layer, rng);
    lic::zero_gradient(device, layer);
    lic::reset_forward_state(device, layer);
    lic::reset_optimizer_state(device, layer, optimizer);
    lic::randn(device, layer.weights.gradient, rng);
    lic::randn(device, layer.weights.gradient_first_order_moment, rng);
    lic::randn(device, layer.weights.gradient_second_order_moment, rng);
    lic::randn(device, layer.biases.gradient, rng);
    lic::randn(device, layer.biases.gradient_first_order_moment, rng);
    lic::randn(device, layer.biases.gradient_second_order_moment, rng);
    lic::increment(layer.weights.parameters, 2, 1, 10);
    lic::increment(layer.weights.gradient, 2, 1, 5);
    lic::increment(layer.weights.gradient_first_order_moment, 2, 1, 2);
    lic::increment(layer.weights.gradient_second_order_moment, 2, 1, 1);
    auto abs_diff = lic::abs_diff(device, layer, layer_1::layer);
    ASSERT_FLOAT_EQ(10 + 5 + 2 + 1, abs_diff);
}

#include "../../../data/test_layer_in_c_nn_models_mlp_persist_code.h"

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = lic::nn_models::mlp::InferenceSpecification<lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    lic::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    lic::malloc(device, mlp);
    lic::init_weights(device, mlp, rng);
    lic::increment(mlp.hidden_layers[0].biases.parameters, 0, 2, 10);
    auto abs_diff = lic::abs_diff(device, mlp, mlp_1::mlp);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP_ADAM){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    using OPTIMIZER_PARAMETERS = lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using SPEC = lic::nn_models::mlp::AdamSpecification<lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>>;
    lic::nn_models::mlp::NeuralNetworkAdam<SPEC> mlp;
    lic::malloc(device, mlp);
    lic::init_weights(device, mlp, rng);
    lic::zero_gradient(device, mlp);
    lic::reset_forward_state(device, mlp);
    lic::reset_optimizer_state(device, mlp, optimizer);
    lic::increment(mlp.hidden_layers[0].biases.parameters, 0, 2, 10);
    mlp_1::mlp.input_layer = mlp_1::input_layer::layer;
    auto abs_diff = lic::abs_diff(device, mlp, mlp_1::mlp);
    ASSERT_FLOAT_EQ(10, abs_diff);
}

TEST(LAYER_IN_C_CONTAINER_PERSIST_CODE_LOAD, TEST_MLP_EVALUATE){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    constexpr typename DEVICE::index_t BATCH_SIZE = 10;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    using STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, typename DEVICE::index_t, 13, 4, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::ActivationFunction::IDENTITY, 1, true, lic::matrix::layouts::RowMajorAlignment<typename DEVICE::index_t, 1>>;
    using SPEC = lic::nn_models::mlp::InferenceSpecification<STRUCTURE_SPEC>;
    lic::nn_models::mlp::NeuralNetwork<SPEC> mlp;
    lic::malloc(device, mlp);
    lic::init_weights(device, mlp, rng);

    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::INPUT_DIM>> input;
    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, BATCH_SIZE, SPEC::STRUCTURE_SPEC::OUTPUT_DIM>> output_orig, output_loaded;
    lic::malloc(device, input);
    lic::malloc(device, output_orig);
    lic::malloc(device, output_loaded);
    lic::randn(device, input, rng);
    lic::evaluate(device, mlp, input, output_orig);
    lic::evaluate(device, mlp_1::mlp, input, output_loaded);
    lic::print(device, output_orig);

    auto output = lic::save(device, input, "input", const_declaration);
    output += lic::save(device, output_orig, "expected_output", const_declaration);

    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_layer_in_c_nn_models_mlp_evaluation.h");
    file << output;
    file.close();

    auto abs_diff = lic::abs_diff(device, output_orig, output_loaded);
    ASSERT_FLOAT_EQ(0, abs_diff);
}