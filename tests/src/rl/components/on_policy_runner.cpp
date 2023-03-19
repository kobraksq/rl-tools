#include <layer_in_c/operations/cpu.h>
#include <layer_in_c/rl/environments/pendulum/pendulum.h>
#include <layer_in_c/rl/environments/pendulum/operations_generic.h>
#include <layer_in_c/nn_models/mlp_unconditional_stddev/operations_cpu.h>
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>
#include <layer_in_c/rl/components/on_policy_runner/operations_generic.h>
#include <layer_in_c/rl/components/on_policy_runner/persist.h>

namespace lic = layer_in_c;


#include <gtest/gtest.h>


TEST(LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER, TEST){
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = lic::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<ENVIRONMENT_SPEC>;

    constexpr TI N_ENVIRONMENTS = 3;
    using ON_POLICY_RUNNER_SPEC = lic::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS>;
    using ON_POLICY_RUNNER = lic::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;


    DEVICE device;
    ON_POLICY_RUNNER runner;
    lic::malloc(device, runner);
    ENVIRONMENT envs[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 199);
    lic::init(device, runner, envs, rng);

    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::TANH>;
    using ACTOR_SPEC = lic::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = lic::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;


    constexpr TI STEPS_PER_ENV = 1000;
    using BUFFER_SPEC = lic::rl::components::on_policy_runner::BufferSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
    using BUFFER = lic::rl::components::on_policy_runner::Buffer<BUFFER_SPEC>;

    ACTOR_TYPE actor;
    ACTOR_BUFFERS actor_buffers;
    BUFFER buffer;
    lic::malloc(device, actor);
    lic::malloc(device, actor_buffers);
    lic::malloc(device, buffer);
    lic::init_weights(device, actor, rng);
    lic::set_all(device, buffer.data, 0);


    lic::collect(device, buffer, runner, actor, actor_buffers, rng);
    lic::print(device, buffer.data);
    lic::collect(device, buffer, runner, actor, actor_buffers, rng);
    lic::print(device, buffer.data);
    lic::collect(device, buffer, runner, actor, actor_buffers, rng);
    lic::print(device, buffer.data);
    ENVIRONMENT::State states[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        states[env_i] = get(runner.states, 0, env_i);
    }
    lic::collect(device, buffer, runner, actor, actor_buffers, rng);
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        for(TI step_i = 0; step_i < BUFFER_SPEC::STEPS_PER_ENV; step_i++){
            TI pos = step_i * ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS + env_i;
            {
                lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
                lic::malloc(device, observation);
                lic::observe(device, get(runner.environments, 0, env_i), states[env_i], observation);
                auto observation_runner = lic::view<DEVICE, decltype(buffer.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, buffer.observations, pos, 0);
                auto abs_diff = lic::abs_diff(device, observation, observation_runner);
                if(!get(buffer.truncated, pos, 0)){
//                    ASSERT_FLOAT_EQ(abs_diff, 0);
                }
                lic::free(device, observation);
            }
            typename ENVIRONMENT::State next_state;
            auto action = lic::view<DEVICE, decltype(buffer.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, buffer.actions, pos, 0);
            step(device, get(runner.environments, 0, env_i), states[env_i], action, next_state);
            states[env_i] = next_state;
        }
    }
    std::string FILE_PATH = "test_rl_components_on_policy_runner_buffer.h5";
    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::Overwrite);
        lic::save(device, buffer, file.createGroup("buffer"));
    }

    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::ReadOnly);
        BUFFER loaded;
        lic::malloc(device, loaded);
        lic::load(device, loaded, file.getGroup("buffer"));
        auto abs_diff = lic::abs_diff(device, loaded.data, buffer.data);
        ASSERT_FLOAT_EQ(0, abs_diff);
    }



}