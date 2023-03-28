#ifndef LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_EXTENSIONS_H
#define LAYER_IN_C_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_EXTENSIONS_H

#include "ppo.h"
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>

namespace layer_in_c{
    namespace rl::algorithms::ppo{

        template <typename PPO_SPEC>
        struct TrainingBuffersHybrid{
            using SPEC = PPO_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI BATCH_SIZE = SPEC::BATCH_SIZE;
            static constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
            static constexpr TI OBSERVATION_DIM = SPEC::ENVIRONMENT::OBSERVATION_DIM;
            Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> actions;
            Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> observations;
            Matrix<matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> d_action_log_prob_d_action;
            Matrix<matrix::Specification<T, TI, BATCH_SIZE, OBSERVATION_DIM>> d_observations;
            Matrix<matrix::Specification<T, TI, BATCH_SIZE, 1>> target_values;
        };
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::ppo::TrainingBuffersHybrid<SPEC>& buffers){
        malloc(device, buffers.actions);
        malloc(device, buffers.observations);
        malloc(device, buffers.d_action_log_prob_d_action);
        malloc(device, buffers.d_observations);
        malloc(device, buffers.target_values);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::algorithms::ppo::TrainingBuffersHybrid<SPEC>& buffers){
        free(device, buffers.actions);
        free(device, buffers.observations);
        free(device, buffers.d_action_log_prob_d_action);
        free(device, buffers.d_observations);
        free(device, buffers.target_values);
    }
    template <typename DEVICE, typename DEVICE_EVALUATION, typename PPO_SPEC, typename OPR_SPEC, auto STEPS_PER_ENV, typename OPTIMIZER, typename RNG>
    void train_hybrid(DEVICE& device,
        DEVICE_EVALUATION& device_evaluation,
        rl::algorithms::PPO<PPO_SPEC>& ppo,
        rl::algorithms::PPO<PPO_SPEC>& ppo_evaluation,
        rl::components::on_policy_runner::Dataset<rl::components::on_policy_runner::DatasetSpecification<OPR_SPEC, STEPS_PER_ENV>>& buffer,
        OPTIMIZER& optimizer,
        rl::algorithms::ppo::Buffers<PPO_SPEC>& ppo_buffers,
        rl::algorithms::ppo::TrainingBuffersHybrid<PPO_SPEC>& hybrid_buffers,
        typename PPO_SPEC::ACTOR_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& actor_buffers,
        typename PPO_SPEC::CRITIC_TYPE::template BuffersForwardBackward<PPO_SPEC::BATCH_SIZE>& critic_buffers,
        RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_ALGORITHMS_PPO_CHECK_INIT
        utils::assert_exit(device, ppo.initialized, "PPO not initialized");
#endif
        using T = typename PPO_SPEC::T;
        using TI = typename PPO_SPEC::TI;
        static_assert(utils::typing::is_same_v<typename PPO_SPEC::ENVIRONMENT, typename OPR_SPEC::ENVIRONMENT>, "environment mismatch");
        using BUFFER = rl::components::on_policy_runner::Dataset<rl::components::on_policy_runner::DatasetSpecification<OPR_SPEC, STEPS_PER_ENV>>;
        static_assert(BUFFER::STEPS_TOTAL > 0);
        constexpr TI N_EPOCHS = PPO_SPEC::PARAMETERS::N_EPOCHS;
        constexpr TI BATCH_SIZE = PPO_SPEC::BATCH_SIZE;
        constexpr TI N_BATCHES = BUFFER::STEPS_TOTAL/BATCH_SIZE;
        static_assert(N_BATCHES > 0);
        constexpr TI ACTION_DIM = OPR_SPEC::ENVIRONMENT::ACTION_DIM;
        constexpr TI OBSERVATION_DIM = OPR_SPEC::ENVIRONMENT::OBSERVATION_DIM;
        // batch needs observations, original log-probs, advantages
        for(TI epoch_i = 0; epoch_i < N_EPOCHS; epoch_i++){
            // shuffling
            for(TI buffer_i = 0; buffer_i < BUFFER::STEPS_TOTAL; buffer_i++){
                TI sample_index = random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), buffer_i, BUFFER::STEPS_TOTAL-1, rng);
                {
                    auto target_row = row(device, buffer.observations, buffer_i);
                    auto source_row = row(device, buffer.observations, sample_index);
                    swap(device, target_row, source_row);
                }
                {
                    auto target_row = row(device, buffer.actions, buffer_i);
                    auto source_row = row(device, buffer.actions, sample_index);
                    swap(device, target_row, source_row);
                }
                swap(device, buffer.advantages      , buffer.advantages      , buffer_i, 0, sample_index, 0);
                swap(device, buffer.action_log_probs, buffer.action_log_probs, buffer_i, 0, sample_index, 0);
                swap(device, buffer.target_values   , buffer.target_values   , buffer_i, 0, sample_index, 0);
            }
            for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
                zero_gradient(device_evaluation, ppo_evaluation.critic);
                zero_gradient(device_evaluation, ppo_evaluation.actor); // has to be reset before accumulating the action-log-std gradient

                auto batch_offset = batch_i * BATCH_SIZE;
                auto batch_observations     = view(device, buffer.observations    , matrix::ViewSpec<BATCH_SIZE, OBSERVATION_DIM>(), batch_offset, 0);
                auto batch_actions          = view(device, buffer.actions         , matrix::ViewSpec<BATCH_SIZE, ACTION_DIM     >(), batch_offset, 0);
                auto batch_action_log_probs = view(device, buffer.action_log_probs, matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);
                auto batch_advantages       = view(device, buffer.advantages      , matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);
                auto batch_target_values    = view(device, buffer.target_values   , matrix::ViewSpec<BATCH_SIZE, 1              >(), batch_offset, 0);

                T advantage_mean = 0;
                T advantage_std = 0;
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T advantage = get(batch_advantages, batch_step_i, 0);
                    advantage_mean += advantage;
                    advantage_std += advantage * advantage;
                }
                advantage_mean /= BATCH_SIZE;
                advantage_std /= BATCH_SIZE;

                advantage_std = math::sqrt(typename DEVICE::SPEC::MATH(), advantage_std - advantage_mean * advantage_mean);
//                add_scalar(device, device.logger, "ppo/advantage/mean", advantage_mean);
//                add_scalar(device, device.logger, "ppo/advantage/std", advantage_std);

                copy(device_evaluation, device, hybrid_buffers.observations, batch_observations);
                forward(device_evaluation, ppo_evaluation.actor, hybrid_buffers.observations, hybrid_buffers.actions);
                copy(device, device_evaluation, ppo_buffers.current_batch_actions, hybrid_buffers.actions);
//                auto abs_diff = abs_diff(device, batch_actions, buffer.actions);

                copy(device, device_evaluation, ppo.actor.log_std.parameters, ppo_evaluation.actor.log_std.parameters);
                copy(device, device_evaluation, ppo.actor.log_std.gradient, ppo_evaluation.actor.log_std.gradient);
                for(TI batch_step_i = 0; batch_step_i < BATCH_SIZE; batch_step_i++){
                    T action_log_prob = 0;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        T current_action = get(ppo_buffers.current_batch_actions, batch_step_i, action_i);
                        T action = get(batch_actions, batch_step_i, action_i);
                        T action_log_std = get(ppo.actor.log_std.parameters, 0, action_i);
                        T action_std = math::exp(typename DEVICE::SPEC::MATH(), action_log_std);
                        T action_diff_by_action_std = (current_action - action) / action_std;
                        action_log_prob += -0.5 * action_diff_by_action_std * action_diff_by_action_std - action_log_std - 0.5 * math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>);
                        set(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, - action_diff_by_action_std / action_std);
//                      d_action_log_prob_d_action_std =  (-action_diff_by_action_std) * (-action_diff_by_action_std)      / action_std - 1 / action_std)
//                      d_action_log_prob_d_action_std = ((-action_diff_by_action_std) * (-action_diff_by_action_std) - 1) / action_std)
//                      d_action_log_prob_d_action_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std
//                      d_action_log_prob_d_action_log_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std * exp(action_log_std)
//                      d_action_log_prob_d_action_log_std = (action_diff_by_action_std * action_diff_by_action_std - 1) / action_std * action_std
//                      d_action_log_prob_d_action_log_std =  action_diff_by_action_std * action_diff_by_action_std - 1
                        T current_entropy = action_log_std + math::log(typename DEVICE::SPEC::MATH(), 2 * math::PI<T>)/(T)2 + (T)1/(T)2;
                        T current_entropy_loss = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT * current_entropy;
                        // todo: think about possible implementation detail: clipping entropy bonus as well (because it changes the distribution)
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d_entropy_loss_d_action_log_std = -(T)1/BATCH_SIZE * PPO_SPEC::PARAMETERS::ACTION_ENTROPY_COEFFICIENT;
                            increment(ppo.actor.log_std.gradient, 0, action_i, current_d_entropy_loss_d_action_log_std);
                            T current_d_action_log_prob_d_action_log_std = action_diff_by_action_std * action_diff_by_action_std - 1;
                            set(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i, current_d_action_log_prob_d_action_log_std);
                        }
                    }
                    T old_action_log_prob = get(batch_action_log_probs, batch_step_i, 0);
                    T advantage = get(batch_advantages, batch_step_i, 0);
                    if(PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
                        advantage = (advantage - advantage_mean) / (advantage_std + PPO_SPEC::PARAMETERS::ADVANTAGE_EPSILON);
                    }
                    T log_ratio = action_log_prob - old_action_log_prob;
                    T ratio = math::exp(typename DEVICE::SPEC::MATH(), log_ratio);
                    // todo: test relative clipping (clipping in log space makes more sense thatn clipping in exp space)
                    T clipped_ratio = math::clamp(typename DEVICE::SPEC::MATH(), ratio, 1 - PPO_SPEC::PARAMETERS::EPSILON_CLIP, 1 + PPO_SPEC::PARAMETERS::EPSILON_CLIP);
                    T normal_advantage = ratio * advantage;
                    T clipped_advantage = clipped_ratio * advantage;
                    T slippage = 0.0;
                    bool ratio_min_switch = normal_advantage - clipped_advantage <= slippage;
                    T clipped_surrogate = ratio_min_switch ? normal_advantage : clipped_advantage;

                    T d_loss_d_clipped_surrogate = -(T)1/BATCH_SIZE;
                    T d_clipped_surrogate_d_ratio = ratio_min_switch ? advantage : 0;
                    T d_ratio_d_action_log_prob = ratio;
                    T d_loss_d_action_log_prob = d_loss_d_clipped_surrogate * d_clipped_surrogate_d_ratio * d_ratio_d_action_log_prob;
                    for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                        multiply(ppo_buffers.d_action_log_prob_d_action, batch_step_i, action_i, d_loss_d_action_log_prob);
                        if(PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
                            T current_d_action_log_prob_d_action_log_std = get(ppo_buffers.d_action_log_prob_d_action_log_std, batch_step_i, action_i);
                            increment(ppo.actor.log_std.gradient, 0, action_i, d_loss_d_action_log_prob * current_d_action_log_prob_d_action_log_std);
                        }
                    }
                }
                copy(device_evaluation, device, ppo_evaluation.actor.log_std.parameters, ppo.actor.log_std.parameters);
                copy(device_evaluation, device, ppo_evaluation.actor.log_std.gradient, ppo.actor.log_std.gradient);

                copy(device_evaluation, device, hybrid_buffers.d_action_log_prob_d_action, ppo_buffers.d_action_log_prob_d_action);
                backward(device_evaluation, ppo_evaluation.actor, hybrid_buffers.observations, hybrid_buffers.d_action_log_prob_d_action, hybrid_buffers.d_observations, actor_buffers);
                copy(device_evaluation, device, hybrid_buffers.target_values, batch_target_values);
                forward_backward_mse(device_evaluation, ppo_evaluation.critic, hybrid_buffers.observations, hybrid_buffers.target_values, critic_buffers);
                update(device_evaluation, ppo_evaluation.actor, optimizer);
                update(device_evaluation, ppo_evaluation.critic, optimizer);
            }
        }
    }

}
#endif