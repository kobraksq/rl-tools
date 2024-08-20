#define MUX
#ifdef RL_TOOLS_ENABLE_TRACY
#include "Tracy.hpp"
#endif
//#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
#ifdef MUX
#include <rl_tools/operations/cpu_mux.h>
#else
#include <rl_tools/operations/cpu.h>
#endif
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#ifdef MUX
#include <rl_tools/nn/operations_cpu_mux.h>
#else
#include <rl_tools/nn/operations_cpu.h>
#endif
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/layers/embedding/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential_v2/persist.h>
#include "dataset.h"

namespace rlt = rl_tools;

#include <chrono>
#include <queue>
#include <algorithm>
#include <filesystem>


#ifdef MUX
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE = rlt::devices::DefaultCPU;
#endif
using TI = typename DEVICE::index_t;
using T = float;

template <typename T, typename TI>
struct Config{
    struct BASE{
        static constexpr TI BATCH_SIZE = 16;
        static constexpr TI INPUT_DIM = 1;
        static constexpr TI OUTPUT_DIM = 1;
        static constexpr TI HIDDEN_DIM = 16;
        static constexpr TI SEQUENCE_LENGTH = 128;
        static constexpr TI HORIZON = 10;
        static constexpr TI DATASET_SIZE = 100000;
    };

    using PARAMS = BASE;
//    using PARAMS = USEFUL;

    template <TI T_BATCH_SIZE>
    using INPUT_SHAPE_TEMPLATE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, T_BATCH_SIZE, 1>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMS::BATCH_SIZE>;
    using INPUT_SHAPE = INPUT_SHAPE_TEMPLATE<PARAMS::BATCH_SIZE>;
    using RESET_SHAPE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, PARAMS::BATCH_SIZE, 1>;
    using RESET_TYPE = rlt::tensor::Specification<bool, TI, RESET_SHAPE>;
    using INPUT_SPEC = rlt::tensor::Specification<T, TI, INPUT_SHAPE>;
    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, PARAMS::SEQUENCE_LENGTH, PARAMS::INPUT_DIM, PARAMS::HIDDEN_DIM, rlt::nn::parameters::Gradient, rlt::TensorDynamicTag, true>;
    using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
    using DENSE_LAYER1_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMS::HIDDEN_DIM, PARAMS::HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, PARAMS::SEQUENCE_LENGTH>>;
    using DENSE_LAYER1_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER1_SPEC>;
    using DENSE_LAYER2_SPEC = rlt::nn::layers::dense::Specification<T, TI, PARAMS::HIDDEN_DIM, PARAMS::OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, PARAMS::SEQUENCE_LENGTH>>;
    using DENSE_LAYER2_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER2_SPEC>;
    using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
    using MODEL = typename IF::template Module<GRU_TEMPLATE:: template Layer, typename IF::template Module<DENSE_LAYER1_TEMPLATE::template Layer, typename IF::template Module<DENSE_LAYER2_TEMPLATE::template Layer>>>;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, PARAMS::BATCH_SIZE, PARAMS::OUTPUT_DIM>;
    using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;
    using OUTPUT_TARGET_SHAPE = rlt::tensor::Shape<TI, PARAMS::SEQUENCE_LENGTH, PARAMS::BATCH_SIZE, 1>;
    using OUTPUT_TARGET_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_TARGET_SHAPE>;
    struct ADAM_PARAMS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
        static constexpr T ALPHA = 0.01;
    };
    using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMS>;
    using ADAM = rlt::nn::optimizers::Adam<ADAM_SPEC>;
};

using CONFIG = Config<T, TI>;

int main(){
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    std::string data_path, file_name;
    std::string dataset_string;

    typename CONFIG::MODEL model;
    typename CONFIG::MODEL::Buffer<CONFIG::PARAMS::BATCH_SIZE> buffer;
    typename CONFIG::ADAM optimizer;
    rlt::Tensor<typename CONFIG::INPUT_SPEC> input;
    rlt::Tensor<typename CONFIG::RESET_TYPE> reset;
    rlt::Tensor<typename CONFIG::OUTPUT_SPEC> d_output;
    rlt::Tensor<typename CONFIG::OUTPUT_TARGET_SPEC> output_target;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, reset);
    rlt::malloc(device, d_output);
    rlt::malloc(device, output_target);
    rlt::init_weights(device, model, rng);
    rlt::reset_optimizer_state(device, optimizer, model);
    constexpr T PROBABILITY = 1 * 1/((T)CONFIG::PARAMS::HORIZON);
    std::cout << "Probability: " << PROBABILITY << std::endl;
    for(TI epoch_i=0; epoch_i < 1000; epoch_i++){
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_print = start_time;
        for(TI sample_i=0; sample_i < CONFIG::PARAMS::DATASET_SIZE; sample_i += CONFIG::PARAMS::BATCH_SIZE){
            for(TI batch_i = 0; batch_i < CONFIG::PARAMS::BATCH_SIZE; batch_i++){
                std::vector<T> values;
                for(TI sequence_i = 0; sequence_i < CONFIG::PARAMS::SEQUENCE_LENGTH; sequence_i++){
                    constexpr bool MAX_VALUE = false;
                    if constexpr(MAX_VALUE){
                        // max value
                        T new_value = rlt::random::normal_distribution::sample(device.random, (T)0, (T)1, rng);
                        values.push_back(new_value);
                        while(values.size() > CONFIG::PARAMS::HORIZON){
                            values.erase(values.begin());
                        }
                        set(device, input, new_value, sequence_i, batch_i, 0);
                        T max_value = *std::max_element(values.begin(), values.end());
                        set(device, output_target, max_value, sequence_i, batch_i, 0);
                        bool reset_now = false; //rlt::random::uniform_real_distribution(device.random, (T)0, (T)1, rng) < 0.1;
                        if(reset_now){
                            values.clear();
                        }
                    }
                    else{
                        // needle in haystack
                        bool reset_now = rlt::random::uniform_real_distribution(device.random, (T)0, (T)1, rng) < 0.5/((T)CONFIG::PARAMS::HORIZON);
                        if(sequence_i == 0 || reset_now){
                            values.clear();
                            rlt::set(device, reset, true, sequence_i, batch_i, 0);
                        }
                        else{
                            rlt::set(device, reset, false, sequence_i, batch_i, 0);
                        }
                        T new_value = rlt::random::normal_distribution::sample(device.random, (T)0, (T)1, rng);
                        if(rlt::random::uniform_real_distribution(device.random, (T)0, (T)1, rng) < PROBABILITY){
                            new_value = 1;
                        }
                        values.push_back(new_value);
                        while(values.size() > CONFIG::PARAMS::HORIZON){
                            values.erase(values.begin());
                        }
                        T number_of_ones = std::count(values.begin(), values.end(), 1);
                        set(device, input, new_value, sequence_i, batch_i, 0);
                        set(device, output_target, (T)number_of_ones/(CONFIG::PARAMS::HORIZON * PROBABILITY), sequence_i, batch_i, 0);
                    }

                }
            }
            using RESET_MODE_SPEC = rlt::nn::layers::gru::ResetModeSpecification<rlt::nn::mode::Default, decltype(reset)>;
            using RESET_MODE = rlt::nn::layers::gru::ResetMode<RESET_MODE_SPEC>;
            rlt::nn::Mode<RESET_MODE> reset_mode;
            reset_mode.reset_container = reset;
            rlt::forward(device, model, input, buffer, rng, reset_mode);
            auto output = rlt::output(device, model);
            auto output_matrix_view = rlt::matrix_view(device, output);
            auto output_target_matrix_view = rlt::matrix_view(device, output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            rlt::nn::loss_functions::mse::gradient(device, output_matrix_view, output_target_matrix_view, d_output_matrix_view);
            T elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            T elapsed_print = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_print).count() / 1000.0;
            if(elapsed_print > 2.0 || sample_i % 1000000 == 0){
                for(TI sequence_i = 0; sequence_i < CONFIG::PARAMS::SEQUENCE_LENGTH; sequence_i++){
                    std::cout << "Step: " << sequence_i << std::endl;
                    for(TI sample_i = 0; sample_i < 1; sample_i++){
                        std::cout << "Input: " << rlt::get(device, input, sequence_i, sample_i, 0) << " Reset: " << rlt::get(device, reset, sequence_i, sample_i, 0) << " Target: " << rlt::get(device, output_target, sequence_i, sample_i, 0) << " => " << rlt::get(device, output, sequence_i, sample_i, 0) << std::endl;
                    }
                }
                T loss = rlt::nn::loss_functions::mse::evaluate(device, output_matrix_view, output_target_matrix_view);
                last_print = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << epoch_i << " Sample: " << sample_i << " Batch: " << sample_i/CONFIG::PARAMS::BATCH_SIZE << " (" << sample_i/CONFIG::PARAMS::BATCH_SIZE/elapsed << " batch/s)" << " Loss: " << loss << std::endl;
            }
            rlt::zero_gradient(device, model);
            rlt::backward(device, model, input, d_output, buffer, reset_mode);
            rlt::step(device, optimizer, model);
        }
    }
    return 0;
}
