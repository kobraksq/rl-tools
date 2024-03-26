#include <gtest/gtest.h>
#include <iostream>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/tensor.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
namespace rlt = rl_tools;

constexpr double EPSILON = 1e-8;

template <typename INPUT>
void test_shape_operations(typename INPUT::TI length){
    ASSERT_TRUE(length == rlt::length(INPUT{}));
    using APPEND = rlt::tensor::Append<INPUT, 5>;
    static_assert((rlt::length(INPUT{}) + 1) == rlt::length(APPEND{}));
    using PREPEND = rlt::tensor::Prepend<INPUT, 10>;
    static_assert((rlt::length(INPUT{}) + 1) == rlt::length(PREPEND{}));

    if constexpr(rlt::length(INPUT{}) > 1){
        using POP_FRONT = rlt::tensor::PopFront<INPUT>;
        static_assert(rlt::length(INPUT{}) == (rlt::length(POP_FRONT{}) + 1));
        using POP_BACK = rlt::tensor::PopBack<INPUT>;
        static_assert(rlt::length(INPUT{}) == (rlt::length(POP_BACK{}) + 1));

        if constexpr (rlt::length(INPUT{}) >= 3){
            static_assert(rlt::get<0>(POP_FRONT{}) == rlt::get<1>(INPUT{}));
            static_assert(rlt::get<1>(POP_FRONT{}) == rlt::get<2>(INPUT{}));
        }
        if constexpr(rlt::length(INPUT{}) >= 3){
            static_assert(rlt::get<0>(POP_BACK{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<1>(POP_BACK{}) == rlt::get<1>(INPUT{}));
        }
        {
            using REMOVE = rlt::tensor::Remove<INPUT, 0>;
            static_assert(rlt::get<0>(REMOVE{}) == rlt::get<1>(INPUT{}));
            static_assert(rlt::length(INPUT{}) - 1 == rlt::length(REMOVE{}));
        }
        {
            using REMOVE = rlt::tensor::Remove<INPUT, 1>;
            static_assert(rlt::length(INPUT{}) - 1 == rlt::length(REMOVE{}));
            static_assert(rlt::get<0>(REMOVE{}) == rlt::get<0>(REMOVE{}));
        }
        if constexpr(rlt::length(INPUT{}) == 2){
            using REMOVE = rlt::tensor::Remove<INPUT, 1>;
            static_assert(rlt::length(REMOVE{}) == 1);
        }
        {
            using INSERT = rlt::tensor::Insert<INPUT, 1337, 0>;
            static_assert(rlt::get<0>(INSERT{}) == 1337);
            static_assert(rlt::length(INPUT{}) + 1 == rlt::length(INSERT{}));
            static_assert(rlt::get<1>(INSERT{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<rlt::length(INPUT{})>(INSERT{}) == rlt::get<rlt::length(INPUT{})-1>(INPUT{}));
        }
        {
            using INSERT = rlt::tensor::Insert<INPUT, 1337, 1>;
            static_assert(rlt::length(INPUT{}) + 1 == rlt::length(INSERT{}));
            static_assert(rlt::get<0>(INSERT{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<1>(INSERT{}) == 1337);
            static_assert(rlt::get<rlt::length(INPUT{})>(INSERT{}) == rlt::get<rlt::length(INPUT{})-1>(INPUT{}));
        }
    }


    if constexpr(rlt::length(INPUT{}) >= 3){
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 0>;
            static_assert(rlt::get<0>(REPLACE{}) == 10);
            static_assert(rlt::get<1>(REPLACE{}) == rlt::get<1>(INPUT{}));
            static_assert(rlt::get<2>(REPLACE{}) == rlt::get<2>(INPUT{}));
        }
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 1>;
            static_assert(rlt::get<1>(REPLACE{}) == 10);
            static_assert(rlt::get<0>(REPLACE{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<2>(REPLACE{}) == rlt::get<2>(INPUT{}));
        }
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 2>;
            static_assert(rlt::get<2>(REPLACE{}) == 10);
            static_assert(rlt::get<0>(REPLACE{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<1>(REPLACE{}) == rlt::get<1>(INPUT{}));
        }
    }
    {

        using REPLACE = rlt::tensor::Replace<INPUT, 10, 0>;
        static_assert(rlt::get<0>(REPLACE{}) == 10);
    }
    {
        using REPLACE = rlt::tensor::Replace<INPUT, 1337, rlt::length(INPUT{})-1>;
        static_assert(rlt::get<rlt::length(INPUT{})-1>(REPLACE{}) == 1337);
    }

    if constexpr(rlt::length(INPUT{}) == 3){
        using PRODUCT = rlt::tensor::Product<INPUT>;
        ASSERT_TRUE(rlt::get<0>(PRODUCT{}) == rlt::get<0>(INPUT{}) * rlt::get<1>(INPUT{}) * rlt::get<2>(INPUT{}));
    }
}

TEST(RL_TOOLS_TENSOR_TEST, SHAPE_OPERATIONS){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3, 4>>(3);
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3>>(2);
    test_shape_operations<rlt::tensor::Shape<TI, 2>>(1);
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0>>(10);
}
TEST(RL_TOOLS_TENSOR_TEST, SHAPE_OPERATIONS2){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    {
        using SHAPE = rlt::tensor::Shape<TI>;
        using INSERT = rlt::tensor::Insert<SHAPE, 1337, 0>;
        static_assert(rlt::get<0>(INSERT{}) == 1337);
    }
    {
        using SHAPE = rlt::tensor::Shape<TI>;
        using APPEND = rlt::tensor::Append<SHAPE, 1337>;
        using INSERT = rlt::tensor::Insert<APPEND, 1338, 0>;
        static_assert(rlt::get<0>(INSERT{}) == 1338);
        static_assert(rlt::get<1>(INSERT{}) == 1337);
        static_assert(rlt::length(SHAPE{}) == 0);
        static_assert(rlt::length(APPEND{}) == 1);
        static_assert(rlt::length(INSERT{}) == 2);
    }
    {
        using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
        using INSERT = rlt::tensor::Insert<SHAPE, 1337, 0>;
        static_assert(rlt::get<0>(INSERT{}) == 1337);
        static_assert(rlt::get<1>(INSERT{}) == 2);
        static_assert(rlt::get<2>(INSERT{}) == 3);
        static_assert(rlt::get<3>(INSERT{}) == 4);
        static_assert(rlt::length(SHAPE{}) == 3);
        static_assert(rlt::length(INSERT{}) == 4);
    }
    {
        using SHAPE = rlt::tensor::Shape<TI, 2>;
        static_assert(rlt::utils::typing::is_same_v<SHAPE::NEXT_ELEMENT::NEXT_ELEMENT, rlt::tensor::FinalElement>);
        using POP = rlt::tensor::PopFront<SHAPE>;
        using POP_BACK = rlt::tensor::PopBack<SHAPE>;
        static_assert(rlt::length(POP{}) == 0);
        static_assert(rlt::length(POP_BACK{}) == 0);
        using REMOVE = rlt::tensor::Remove<SHAPE, 0>;
        static_assert(rlt::length(SHAPE{}) == 1);
        static_assert(rlt::length(REMOVE{}) == 0);
    }
    {
        using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4, 5>;
        static_assert(rlt::length(SHAPE{}) == 4);
        static_assert(rlt::get<0>(SHAPE{}) == 2);
        static_assert(rlt::get<1>(SHAPE{}) == 3);
        static_assert(rlt::get<2>(SHAPE{}) == 4);
        static_assert(rlt::get<3>(SHAPE{}) == 5);
        using POP = rlt::tensor::PopFront<SHAPE>;
        static_assert(rlt::length(POP{}) == 3);
        static_assert(rlt::get<0>(POP{}) == 3);
        static_assert(rlt::get<1>(POP{}) == 4);
        static_assert(rlt::get<2>(POP{}) == 5);
        using POP_BACK = rlt::tensor::PopBack<POP>;
        static_assert(rlt::length(POP_BACK{}) == 2);
        static_assert(rlt::get<0>(POP_BACK{}) == 3);
        static_assert(rlt::get<1>(POP_BACK{}) == 4);
        using APPEND = rlt::tensor::Append<POP_BACK, 6>;
        using REMOVE = rlt::tensor::Remove<APPEND, 0>;
        static_assert(rlt::length(REMOVE{}) == 2);
        static_assert(rlt::get<0>(REMOVE{}) == 4);
        static_assert(rlt::get<1>(REMOVE{}) == 6);
        using INSERT = rlt::tensor::Insert<REMOVE, 1337, 0>;
        static_assert(rlt::length(INSERT{}) == 3);
        static_assert(rlt::get<0>(INSERT{}) == 1337);
        static_assert(rlt::get<1>(INSERT{}) == 4);
        static_assert(rlt::get<2>(INSERT{}) == 6);
        using REPLACE = rlt::tensor::Replace<INSERT, 1338, 0>;
        static_assert(rlt::length(REPLACE{}) == 3);
        static_assert(rlt::get<0>(REPLACE{}) == 1338);
        static_assert(rlt::get<1>(REPLACE{}) == 4);
        static_assert(rlt::get<2>(REPLACE{}) == 6);
        using REPLACE2 = rlt::tensor::Replace<REPLACE, 1339, rlt::length(REPLACE{}) - 1>;
        static_assert(rlt::length(REPLACE2{}) == 3);
        static_assert(rlt::get<0>(REPLACE2{}) == 1338);
        static_assert(rlt::get<1>(REPLACE2{}) == 4);
        static_assert(rlt::get<2>(REPLACE2{}) == 1339);
    }
}

TEST(RL_TOOLS_TENSOR_TEST, STRIDE){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    std::cout << "dim[0]: " << rlt::get<0>(SHAPE{}) << " stride[0]: " << rlt::get<0>(STRIDE{}) << std::endl;
    std::cout << "dim[1]: " << rlt::get<1>(SHAPE{}) << " stride[1]: " << rlt::get<1>(STRIDE{}) << std::endl;
    std::cout << "dim[2]: " << rlt::get<2>(SHAPE{}) << " stride[2]: " << rlt::get<2>(STRIDE{}) << std::endl;
}

TEST(RL_TOOLS_TENSOR_TEST, MALLOC){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
    DEVICE device;
    rlt::malloc(device, tensor);
}

TEST(RL_TOOLS_TENSOR_TEST, SET){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
    DEVICE device;
    rlt::malloc(device, tensor);
    rlt::set(device, tensor, 1337, 0, 0, 0);
    rlt::set(device, tensor, 1337, 1, 2, 0);
    rlt::print(device, tensor);
}

TEST(RL_TOOLS_TENSOR_TEST, VIEW){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
    DEVICE device;
    rlt::malloc(device, tensor);
    rlt::set(device, tensor, 1337, 0, 0, 0);
    rlt::set(device, tensor, 1337, 1, 2, 0);
    rlt::print(device, tensor);
    {
        auto view = rlt::view(device, tensor, 1);
        rlt::print(device, view);
        rlt::set(device, view, 1338, 2, 0);
        ASSERT_EQ(rlt::get(device, tensor, 1, 2, 0), 1338);
    }

    {
        constexpr TI DIM = 1;
        for (TI i = 0; i < rlt::get<DIM>(SHAPE{}); i++){
            auto view2 = rlt::view(device, tensor, i, rlt::tensor::ViewSpec<DIM>{});
            ASSERT_EQ(rlt::data(tensor) + i * rlt::get<DIM>(STRIDE{}), rlt::data(view2));
            rlt::set(device, view2, (T)(DIM * i), 1, 0);
            switch(DIM){}
            ASSERT_EQ(rlt::get(device, tensor, 1, i, 0), (T)(DIM*i));
        }
    }

    {
        constexpr TI DIM = 2;
        for (TI i = 0; i < rlt::get<DIM>(SHAPE{}); i++){
            auto view2 = rlt::view(device, tensor, i, rlt::tensor::ViewSpec<DIM>{});
            ASSERT_EQ(rlt::data(tensor) + i * rlt::get<DIM>(STRIDE{}), rlt::data(view2));
            rlt::set(device, view2, (T)(DIM * i), 1, 0);
            switch(DIM){}
            ASSERT_EQ(rlt::get(device, tensor, 1, 0, i), (T)(DIM*i));
        }
    }
    std::cout << "afterwards: " << std::endl;
    rlt::print(device, tensor);
}

TEST(RL_TOOLS_TENSOR_TEST, RANDN){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 1);
    {
        using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
        using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
        rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
        rlt::malloc(device, tensor);
        rlt::randn(device, tensor, rng);
        rlt::print(device, tensor);
        rlt::free(device, tensor);
    }
    {
        using SHAPE = rlt::tensor::Shape<TI, 20, 30, 40>;
        using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
        rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
        rlt::malloc(device, tensor);
        rlt::randn(device, tensor, rng);
//        rlt::print(device, tensor);
        T sum = rlt::sum(device, tensor);
        std::cout << "sum: " << sum << std::endl;
        T num_elements = rlt::get<0>(rlt::tensor::Product<SHAPE>{});
        ASSERT_EQ(num_elements, 20*30*40);
        ASSERT_LT(rlt::math::abs(device.math, sum), rlt::math::sqrt(device.math, num_elements) * 5);
        rlt::free(device, tensor);
    }
}
TEST(RL_TOOLS_TENSOR_TEST, COMPARE_DIMS) {
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;
    {
        using SHAPE1 = rlt::tensor::Shape<TI, 20, 30, 40>;
        using SHAPE2 = rlt::tensor::Shape<TI, 20, 30, 40>;
        using SHAPE3 = rlt::tensor::Shape<TI, 10, 31, 41>;
        static_assert(rlt::length(rlt::tensor::PopFront<rlt::tensor::PopFront<rlt::tensor::PopFront<SHAPE1>>>{}) == 0);
        static_assert(rlt::utils::typing::is_same_v<SHAPE1::NEXT_ELEMENT::NEXT_ELEMENT::NEXT_ELEMENT::NEXT_ELEMENT, rlt::tensor::FinalElement>);
        static_assert(rlt::tensor::_same_dimensions_shape<SHAPE1, SHAPE2>());
        static_assert(!rlt::tensor::_same_dimensions_shape<SHAPE1, SHAPE3>());
    }
}
TEST(RL_TOOLS_TENSOR_TEST, SET_ALL) {
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 1);
    using SHAPE = rlt::tensor::Shape<TI, 20, 30, 40>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
    rlt::malloc(device, tensor);
    rlt::randn(device, tensor, rng);
    rlt::set_all(device, tensor, 1337);
    for(TI dim0=0; dim0 < rlt::get<0>(SHAPE{}); dim0++){
        for(TI dim1=0; dim1 < rlt::get<1>(SHAPE{}); dim1++){
            for(TI dim2=0; dim2 < rlt::get<2>(SHAPE{}); dim2++){
                ASSERT_EQ(rlt::get(device, tensor, dim0, dim1, dim2), 1337);
            }
        }
    }
}
TEST(RL_TOOLS_TENSOR_TEST, SUBTRACT) {
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 20, 30, 40>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor, tensor2, diff, abs_diff;
    DEVICE device;
    rlt::malloc(device, tensor);
    rlt::malloc(device, tensor2);
    rlt::malloc(device, diff);
    rlt::malloc(device, abs_diff);
    rlt::set_all(device, tensor, 1337);
    rlt::set_all(device, tensor2, 1338);
    rlt::subtract(device, tensor, tensor2, diff);
    T sum = rlt::sum(device, diff);
    ASSERT_EQ(sum, -(T)rlt::get<0>(rlt::tensor::Product<SHAPE>{}));
    rlt::copy(device, device, diff, abs_diff);
    rlt::abs(device, abs_diff);
    T sum_abs = rlt::sum(device, abs_diff);
    ASSERT_EQ(sum_abs, rlt::get<0>(rlt::tensor::Product<SHAPE>{}));
}
TEST(RL_TOOLS_TENSOR_TEST, COPY) {

    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 1);
    {
        using SHAPE = rlt::tensor::Shape<TI, 20, 30, 40>;
        using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
        rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor, tensor_target, tensor_target2, diff;
        rlt::malloc(device, tensor);
        rlt::malloc(device, tensor_target);
        rlt::malloc(device, tensor_target2);
        rlt::malloc(device, diff);
        rlt::randn(device, tensor, rng);

        for(TI i=0; i < rlt::get<1>(SHAPE{})/2; i++){
            auto view = rlt::view(device, tensor, i, rlt::tensor::ViewSpec<1>{});
            auto view_target = rlt::view(device, tensor_target, rlt::get<1>(SHAPE{})/2 + i, rlt::tensor::ViewSpec<1>{});
            rlt::copy(device, device, view, view_target);
        }

        for(TI i=rlt::get<1>(SHAPE{})/2; i < rlt::get<1>(SHAPE{}); i++){
            auto view = rlt::view(device, tensor, i, rlt::tensor::ViewSpec<1>{});
            auto view_target = rlt::view(device, tensor_target, i-rlt::get<1>(SHAPE{})/2, rlt::tensor::ViewSpec<1>{});
            rlt::copy(device, device, view, view_target);
        }

        for(TI i=0; i < rlt::get<1>(SHAPE{})/2; i++){
            auto view = rlt::view(device, tensor_target, i, rlt::tensor::ViewSpec<1>{});
            auto view_target = rlt::view(device, tensor_target2, rlt::get<1>(SHAPE{})/2 + i, rlt::tensor::ViewSpec<1>{});
            rlt::copy(device, device, view, view_target);
        }

        for(TI i=rlt::get<1>(SHAPE{})/2; i < rlt::get<1>(SHAPE{}); i++){
            auto view = rlt::view(device, tensor_target, i, rlt::tensor::ViewSpec<1>{});
            auto view_target = rlt::view(device, tensor_target2, i-rlt::get<1>(SHAPE{})/2, rlt::tensor::ViewSpec<1>{});
            rlt::copy(device, device, view, view_target);
        }

        rlt::subtract(device, tensor, tensor_target2, diff);
        rlt::abs(device, diff);
        T abs_diff = rlt::sum(device, diff);

        std::cout << "Abs diff: " << abs_diff << std::endl;
        ASSERT_LT(abs_diff, EPSILON);


        rlt::free(device, tensor);
        rlt::free(device, tensor_target);
    }
}
