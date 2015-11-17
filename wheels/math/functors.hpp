#pragma once

#include <cmath>
#include <amp.h>

namespace wheels {

    namespace processor {
        struct cpu { constexpr cpu() noexcept {} };
        struct gpu { constexpr gpu() restrict(amp) {} };
    }


    // forward
    template <class T>
    constexpr T && forward(processor::cpu, std::remove_reference_t<T> & arg) noexcept {
        return (static_cast<T &&>(arg));
    }
    template <class T>
    constexpr T && forward(processor::cpu, std::remove_reference_t<T> && arg) noexcept {
        static_assert(!std::is_lvalue_reference<T>::value, "bad forward call");
        return (static_cast<T &&>(arg));
    }

    template <class T>
    constexpr T && forward(processor::gpu, std::remove_reference_t<T> & arg) restrict(amp) {
        return (static_cast<T &&>(arg));
    }
    template <class T>
    constexpr T && forward(processor::gpu, std::remove_reference_t<T> && arg) restrict(amp) {
        static_assert(!std::is_lvalue_reference<T>::value, "bad forward call");
        return (static_cast<T &&>(arg));
    }


    // functors
#define WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(op, name) \
    template <class = processor::cpu> \
    struct functor_##name { \
        template <class A, class B> \
        constexpr auto operator ()(A && a, B && b) const { \
            return forward<A>(processor::cpu(), a) op forward<B>(processor::cpu(), b); \
        } \
    }; \
    template <> \
    struct functor_##name<processor::gpu> { \
        template <class A, class B> \
        constexpr auto operator ()(A && a, B && b) const restrict(amp) { \
            return forward<A>(processor::gpu(), a) op forward<B>(processor::gpu(), b); \
        } \
    };

    WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(+, plus)
    WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(-, minus)
    WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(*, mul)
    WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(/, div)
    WHEELS_DEFINE_BINARY_OP_FOR_CPU_GPU(%, mod)

}