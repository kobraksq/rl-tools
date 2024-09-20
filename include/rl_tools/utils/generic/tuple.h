#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_GENERIC_TUPLE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_GENERIC_TUPLE_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace utils{
        template<typename T_TI, typename... Types>
        struct Tuple {
            using TI = T_TI;
        };

        template<typename T_TI>
        struct Tuple<T_TI> {
        };

        template<typename T_TI, typename Head, typename... Tail>
        struct Tuple<T_TI, Head, Tail...> : Tuple<T_TI, Tail...> {
            using NEXT = Tuple<T_TI, Tail...>;
            Head content;
        };

        template<typename TUPLE, template<typename> typename F>
        struct MapTuple;

        template<typename T_TI, template<typename> typename F>
        struct MapTuple<Tuple<T_TI>, F> {
            using type = Tuple<T_TI>;
        };


        template<typename T_TI, typename Head, typename... Tail, template<typename> typename F>
        struct MapTuple<Tuple<T_TI, Head, Tail...>, F> : MapTuple<Tuple<T_TI, Tail...>, F> {
            using CONTENT = typename F<Head>::CONTENT;
            CONTENT content;
        };
    }
    template<typename TI>
    constexpr TI length(utils::Tuple<TI> &tuple) {
        return 1;
    }

    template<typename TI, typename CURRENT_TYPE, typename... Types>
    constexpr TI length(utils::Tuple<TI, CURRENT_TYPE, Types...> &tuple) {
        return 1 + length(static_cast<utils::Tuple<TI, Types...> &>(tuple));
    }

    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types>
    auto &get(utils::Tuple<TI, CURRENT_TYPE, Types...> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<utils::Tuple<TI, Types...> &>(tuple));
        }
    }

    template<auto INDEX, typename TI, typename CURRENT_TYPE, typename... Types, template <typename> typename F>
    auto &get(utils::MapTuple<utils::Tuple<TI, CURRENT_TYPE, Types...>, F> &tuple) {
        if constexpr (INDEX == 0) {
            return tuple.content;
        } else {
            return get<INDEX - 1>(static_cast<utils::MapTuple<utils::Tuple<TI, Types...>, F> &>(tuple));
        }
    }


    template <typename DEVICE, typename T_TI>
    void malloc(DEVICE& device, utils::Tuple<T_TI>& tuple){ }
    template <typename DEVICE, typename T_TI, typename CURRENT_TYPE, typename... Types>
    void malloc(DEVICE& device, utils::Tuple<T_TI, CURRENT_TYPE, Types...>& tuple){
        malloc(device, tuple.content);
        if constexpr(sizeof...(Types) > 0){
            malloc(device, static_cast<utils::Tuple<T_TI, Types...>&>(tuple));
        }
    }
    template <typename DEVICE, typename T_TI, template <typename> typename F>
    void malloc(DEVICE& device, utils::MapTuple<utils::Tuple<T_TI>, F>& tuple){ }
    template <typename DEVICE, typename T_TI, template <typename> typename F, typename CURRENT_TYPE, typename... Types>
    void malloc(DEVICE& device, utils::MapTuple<utils::Tuple<T_TI, CURRENT_TYPE, Types...>, F>& tuple){
        malloc(device, tuple.content);
        if constexpr(sizeof...(Types) > 0){
            malloc(device, static_cast<utils::MapTuple<utils::Tuple<T_TI, Types...>, F>&>(tuple));
        }
    }

    template <typename DEVICE, typename T_TI>
    void free(DEVICE& device, utils::Tuple<T_TI>& tuple){ }
    template <typename DEVICE, typename T_TI, typename CURRENT_TYPE, typename... Types>
    void free(DEVICE& device, utils::Tuple<T_TI, CURRENT_TYPE, Types...>& tuple){
        free(device, tuple.content);
        if constexpr(sizeof...(Types) > 0){
            free(device, static_cast<utils::Tuple<T_TI, Types...>&>(tuple));
        }
    }
    template <typename DEVICE, typename T_TI, template <typename> typename F>
    void free(DEVICE& device, utils::MapTuple<utils::Tuple<T_TI>, F>& tuple){ }
    template <typename DEVICE, typename T_TI, template <typename> typename F, typename CURRENT_TYPE, typename... Types>
    void free(DEVICE& device, utils::MapTuple<utils::Tuple<T_TI, CURRENT_TYPE, Types...>, F>& tuple){
        free(device, tuple.content);
        if constexpr(sizeof...(Types) > 0){
            free(device, static_cast<utils::MapTuple<utils::Tuple<T_TI, Types...>, F>&>(tuple));
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif