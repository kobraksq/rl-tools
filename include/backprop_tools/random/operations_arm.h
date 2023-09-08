#include "../version.h"
#if !defined(BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_ARM_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_UTILS_RANDOM_OPERATIONS_ARM_H


#include "../utils/generic/typing.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::random{
   devices::random::ARM::index_t default_engine(const devices::random::ARM& dev, devices::random::ARM::index_t seed = 1){
       return 0b10101010101010101010101010101010 + seed;
   };
   constexpr devices::random::ARM::index_t next_max(const devices::random::ARM& dev){
       return devices::random::ARM::MAX_INDEX;
   }
   template<typename RNG>
   void next(const devices::random::ARM& dev, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       rng ^= (rng << 13);
       rng ^= (rng >> 17);
       rng ^= (rng << 5);
   }

   template<typename T, typename RNG>
   T uniform_int_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       using TI = devices::random::ARM::index_t;
       TI range = static_cast<devices::random::ARM::index_t>(high - low) + 1;
       next(dev, rng);
       TI r = rng % range;
       return static_cast<T>(r) + low;
   }
   template<typename T, typename RNG>
   T uniform_real_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
       static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
       static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
       next(dev, rng);
       return (rng / static_cast<T>(next_max(dev))) * (high - low) + low;
   }
    namespace normal_distribution{
        template<typename T, typename RNG>
        T sample(const devices::random::ARM& dev, T mean, T std, RNG& rng){
            static_assert(utils::typing::is_same_v<RNG, devices::random::ARM::index_t>);
            static_assert(utils::typing::is_same_v<T, double> || utils::typing::is_same_v<T, float>);
            next(dev, rng);
            T u1 = rng / static_cast<T>(next_max(dev));
            next(dev, rng);
            T u2 = rng / static_cast<T>(next_max(dev));
            T x = math::sqrt(devices::math::ARM(), -2.0 * math::log(devices::math::ARM(), u1));
            T y = 2.0 * math::PI<T> * u2;
            T z = x * math::cos(devices::math::ARM(), y);
            return z * std + mean;
        }
        template<typename DEVICE, typename T>
        T log_prob(const devices::random::ARM& dev, T mean, T log_std, T value){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            T neg_log_sqrt_pi = -0.5 * math::log(typename DEVICE::SPEC::MATH{}, 2 * math::PI<T>);
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return neg_log_sqrt_pi - log_std - 0.5 * pre_square * pre_square;
        }
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
