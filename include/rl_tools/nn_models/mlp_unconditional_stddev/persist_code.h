#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_PERSIST_CODE_H



#include <string>
#include <sstream>
#include "../../persist/code.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, template <typename> typename BASE>
    persist::Code save_code_split(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        std::stringstream indent_ss;
        for(TI i=0; i < indent; i++){
            indent_ss << "    ";
        }
        std::string ind = indent_ss.str();
        std::stringstream ss_header;
        ss_header << "#include <rl_tools/nn_models/mlp_unconditional_stddev/network.h>\n";
        std::stringstream ss;
        ss << ind << "namespace " << name << " {\n";
        auto input_layer = save_code_split(device, network.input_layer, "input_layer", const_declaration, indent+1);
        ss_header << input_layer.header;
        ss << input_layer.body;
        for(TI hidden_layer_i = 0; hidden_layer_i < SPEC::NUM_HIDDEN_LAYERS; hidden_layer_i++){
            auto hidden_layer = save_code_split(device, network.hidden_layers[hidden_layer_i], "hidden_layer_" + std::to_string(hidden_layer_i), const_declaration, indent+1);
            ss_header << hidden_layer.header;
            ss << hidden_layer.body;
        }
        auto output_layer = save_code_split(device, network.output_layer, "output_layer", const_declaration, indent+1);
        ss_header << output_layer.header;
        ss << output_layer.body;
        ss << ind << "    using SPEC = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::Specification<";
        ss << containers::persist::get_type_string<T>() << ", ";
        ss << containers::persist::get_type_string<TI>() << ", ";
        ss << SPEC::INPUT_DIM << ", " << SPEC::OUTPUT_DIM << ", " << SPEC::NUM_LAYERS << ", " << SPEC::HIDDEN_DIM << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<SPEC::HIDDEN_ACTIVATION_FUNCTION>() << ", ";
        ss << nn::layers::dense::persist::get_activation_function_string<SPEC::OUTPUT_ACTIVATION_FUNCTION>() << ", ";
        ss << ind << "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag, true, RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<" << containers::persist::get_type_string<TI>() << ", 1>>; \n";
        ss << ind << "    template <typename CAPABILITY>" << "\n";
        ss << ind << "    using TEMPLATE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::NeuralNetwork<CAPABILITY, SPEC>; \n";
        ss << ind << "    using CAPABILITY = " << to_string(typename SPEC::CAPABILITY{}) << "; \n";
        ss << ind << "    using TYPE = RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn_models::mlp::NeuralNetwork<CAPABILITY, SPEC>; \n";
        ss << ind << "    " << (const_declaration ? "const " : "") << "TYPE module = {";
        ss << "input_layer::module, ";
        ss << "{";
        for(TI hidden_layer_i = 0; hidden_layer_i < SPEC::NUM_HIDDEN_LAYERS; hidden_layer_i++){
            if(hidden_layer_i > 0){
                ss << ", ";
            }
            ss << "hidden_layer_" << hidden_layer_i << "::module";
        }
        ss << "}, ";
        ss << "output_layer::module";
        ss << "};\n";

        ss << ind << "}\n";
        return {ss_header.str(), ss.str()};
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE>
    persist::Code save_code_split(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
        return save_code_split(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>&>(network), name, const_declaration, indent);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE>
    persist::Code save_code_split(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network, std::string name, bool const_declaration=false, typename DEVICE::index_t indent = 0){
        return save_code_split(device, static_cast<nn_models::mlp_unconditional_stddev::NeuralNetworkBackward<SPEC, BASE>&>(network), name, const_declaration, indent);
    }
    template<typename DEVICE, typename SPEC>
    std::string save_code(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC>& network, std::string name, bool const_declaration = true, typename DEVICE::index_t indent = 0) {
        auto code = save_code_split(device, network, name, const_declaration, indent);
        return code.header + code.body;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
