#pragma once

#include <cmath>
#include <amp.h>

#include "tensor_shape.hpp"

namespace wheels {

   


    struct platform_cpu {
        static constexpr int id = 1;

        // forward
        template <class T>
        static constexpr T && forward(std::remove_reference_t<T> & arg) noexcept {
            return (static_cast<T &&>(arg));
        }
        template <class T>
        static constexpr T && forward(std::remove_reference_t<T> && arg) noexcept {
            static_assert(!std::is_lvalue_reference<T>::value, "bad forward call");
            return (static_cast<T &&>(arg));
        }

        // for_each
        template <class ShapeT, class FunT>
        static constexpr void for_each(const ShapeT & shape, const FunT & fun) {

        }


        // functors
#define WHEELS_DEFINE_BINARY_OP_FOR_CPU(op, name) \
        template <class A, class B> \
        static constexpr auto name(A && a, B && b) { \
            return forward<A>(a) op forward<B>(b); \
        }

        WHEELS_DEFINE_BINARY_OP_FOR_CPU(+, plus)
        WHEELS_DEFINE_BINARY_OP_FOR_CPU(-, minus)
        WHEELS_DEFINE_BINARY_OP_FOR_CPU(*, mul)
        WHEELS_DEFINE_BINARY_OP_FOR_CPU(/, div)
        WHEELS_DEFINE_BINARY_OP_FOR_CPU(%, mod)

    };




    namespace details {

        template <class T, class ... SizeTs, size_t ... Is>
        inline concurrency::extent<sizeof...(SizeTs)>
            _shape_to_extent_seq(const tensor_shape<T, SizeTs ...> & s,
                std::index_sequence<Is...>) {
            return concurrency::extent<sizeof...(SizeTs)>(s.at(const_index<Is>()) ...);
        }

        template <class FunT>
        inline void _invoke_with_amp_index(concurrency::index<1> idx, const FunT & fun) restrict(amp) {
            fun(idx[0]);
        }
        template <class FunT>
        inline void _invoke_with_amp_index(concurrency::index<2> idx, const FunT & fun) restrict(amp) {
            fun(idx[0], idx[1]);
        }
        template <class FunT>
        inline void _invoke_with_amp_index(concurrency::index<3> idx, const FunT & fun) restrict(amp) {
            fun(idx[0], idx[1], idx[2]);
        }

    }


    struct platform_gpu {
        static constexpr int id = 2;

        // forward
        template <class T>
        static constexpr T && forward(std::remove_reference_t<T> & arg) restrict(cpu, amp) {
            return (static_cast<T &&>(arg));
        }
        template <class T>
        static constexpr T && forward(std::remove_reference_t<T> && arg) restrict(cpu, amp) {
            static_assert(!std::is_lvalue_reference<T>::value, "bad forward call");
            return (static_cast<T &&>(arg));
        }       


        // for_each
        template <class ShapeT, class FunT>
        static void for_each(const ShapeT & shape, FunT fun) {
            constexpr int degree = (int)ShapeT::degree();
            concurrency::extent<degree> ext = details::_shape_to_extent_seq(shape, std::make_index_sequence<degree>());
            concurrency::parallel_for_each(ext, [fun](concurrency::index<degree> idx) restrict(amp) {
                details::_invoke_with_amp_index(idx, fun);
            });
        }


        // functors
#define WHEELS_DEFINE_BINARY_OP_FOR_GPU(op, name) \
        template <class A, class B> \
        static constexpr auto name(A && a, B && b) restrict(cpu, amp) { \
            return forward<A>(a) op forward<B>(b); \
        }

        WHEELS_DEFINE_BINARY_OP_FOR_GPU(+, plus)
        WHEELS_DEFINE_BINARY_OP_FOR_GPU(-, minus)
        WHEELS_DEFINE_BINARY_OP_FOR_GPU(*, mul)
        WHEELS_DEFINE_BINARY_OP_FOR_GPU(/, div)
        WHEELS_DEFINE_BINARY_OP_FOR_GPU(%, mod)

    };



    // platform_t
    namespace details {
        template <int PlatformID> struct _platform_of_id {};
        template <> struct _platform_of_id<platform_cpu::id> { using type = platform_cpu; };
        template <> struct _platform_of_id<platform_gpu::id> { using type = platform_gpu; };
    }
    template <int PlatformID> 
    using platform_t = typename details::_platform_of_id<PlatformID>::type;



}