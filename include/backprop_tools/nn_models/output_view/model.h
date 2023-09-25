#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_NN_MODELS_OUTPUT_VIEW_MODEL_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_NN_MODELS_OUTPUT_VIEW_MODEL_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn_models::output_view{
    template <typename T_TI, typename T_MODEL, T_TI T_OFFSET, T_TI T_DIM>
    struct MODEL_VIEW_SPEC{
        using TI = T_TI;
        using MODEL = T_MODEL;
        using T = typename MODEL::T;
        constexpr static TI OFFSET = T_OFFSET;
        constexpr static TI DIM = T_DIM;
        static_assert(MODEL::OUTPUT_DIM >= OFFSET + DIM);
    };
    template <typename T_SPEC>
    struct MODEL{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::MODEL::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::DIM;
        typename SPEC::MODEL& model;
        MODEL(typename SPEC::MODEL& model): model(model){}
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFERS>
    void evaluate(DEVICE& device, const nn_models::output_view::MODEL<SPEC>& actor, Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, BUFFERS& eval_buffers){
        using T = typename OUTPUT_SPEC::T;
        using TI = typename OUTPUT_SPEC::TI;
        static_assert(OUTPUT_SPEC::COLS == SPEC::DIM);
        static_assert(BUFFERS::BATCH_SIZE == SPEC::DIM);
        MatrixDynamic<matrix::Specification<T, TI, OUTPUT_SPEC::ROWS, SPEC::MODEL::OUTPUT_DIM>> actor_output;
        malloc(device, actor_output);
        evaluate(device, actor.model, input, actor_output, eval_buffers);
        auto output_view = view(device, actor_output, matrix::ViewSpec<OUTPUT_SPEC::ROWS, SPEC::DIM>{}, 0, SPEC::OFFSET);
        copy(device, device, output, output_view);
        free(device, actor_output);
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END


#endif