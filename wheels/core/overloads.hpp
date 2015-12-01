#pragma once

#include "constants.hpp"

namespace wheels {

    // overload operators without losing any type information

    template <class T>
    struct join_overloading : no {};

    template <class T>
    struct info_for_overloading {
        using type = T;
    };
    template <class T>
    using info_for_overloading_t = typename info_for_overloading<T>::type;


    // the overloaded<...> functor is called 
    // if any of the parameters join overloading
    template <class OpT, class ... ArgInfoTs>
    struct overloaded {
        constexpr overloaded() {}
        template <class ... ArgTs>
        void operator()(ArgTs && ...) const {
            static_assert(always<bool, false, ArgTs ...>::value,
                "error: this overloaded operator is not implemented, "
                "instantiate overloaded<...> to fix this.");
        }
        template <class Archive>
        void serialize(Archive &) {}
    };



#define WHEELS_OVERLOAD_UNARY_OP(op, name) \
    struct unary_op_##name { \
        constexpr unary_op_##name() {} \
        template <class TT> \
        constexpr decltype(auto) operator()(TT && v) const {\
            return op forward<TT>(v); \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class T, \
        class = std::enable_if_t<join_overloading<std::decay_t<T>>::value>> \
    constexpr decltype(auto) operator op (T && v) { \
        return overloaded<unary_op_##name, info_for_overloading_t<std::decay_t<T>>>()(forward<T>(v)); \
    }

    WHEELS_OVERLOAD_UNARY_OP(-, minus)
    WHEELS_OVERLOAD_UNARY_OP(!, not)
    WHEELS_OVERLOAD_UNARY_OP(~, bitwise_not)


#define WHEELS_OVERLOAD_BINARY_OP(op, name) \
    struct binary_op_##name { \
        constexpr binary_op_##name() {} \
        template <class TT1, class TT2> \
        constexpr decltype(auto) operator()(TT1 && v1, TT2 && v2) const {\
            return (forward<TT1>(v1) op forward<TT2>(v2)); \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class T1, class T2, class = std::enable_if_t< \
        join_overloading<std::decay_t<T1>>::value || \
        join_overloading<std::decay_t<T2>>::value>> \
    constexpr decltype(auto) operator op (T1 && v1, T2 && v2) { \
        return overloaded<binary_op_##name, \
            info_for_overloading_t<std::decay_t<T1>>, \
            info_for_overloading_t<std::decay_t<T2>> \
        >()(forward<T1>(v1), forward<T2>(v2)); \
    }

    WHEELS_OVERLOAD_BINARY_OP(+, plus)
    WHEELS_OVERLOAD_BINARY_OP(-, minus)
    WHEELS_OVERLOAD_BINARY_OP(*, mul)
    WHEELS_OVERLOAD_BINARY_OP(/ , div)
    WHEELS_OVERLOAD_BINARY_OP(%, mod)

    WHEELS_OVERLOAD_BINARY_OP(== , eq)
    WHEELS_OVERLOAD_BINARY_OP(!= , neq)
    WHEELS_OVERLOAD_BINARY_OP(<, lt)
    WHEELS_OVERLOAD_BINARY_OP(<= , lte)
    WHEELS_OVERLOAD_BINARY_OP(>, gt)
    WHEELS_OVERLOAD_BINARY_OP(>= , gte)

    WHEELS_OVERLOAD_BINARY_OP(&&, and)
    WHEELS_OVERLOAD_BINARY_OP(|| , or )
    WHEELS_OVERLOAD_BINARY_OP(&, bitwise_and)
    WHEELS_OVERLOAD_BINARY_OP(| , bitwise_or)
    WHEELS_OVERLOAD_BINARY_OP(^, bitwise_xor)



#define WHEELS_OVERLOAD_UNARY_FUNC(name) \
    struct unary_func_##name { \
        constexpr unary_func_##name() {} \
        template <class TT> \
        constexpr decltype(auto) operator()(TT && v) const {\
            using std::name;\
            return name(forward<TT>(v)); \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class T, \
        class = std::enable_if_t<join_overloading<std::decay_t<T>>::value>, \
        class = void> \
    constexpr decltype(auto) name(T && v) { \
        return overloaded<unary_func_##name, info_for_overloading_t<std::decay_t<T>>>()(forward<T>(v)); \
    }

    WHEELS_OVERLOAD_UNARY_FUNC(sin)
    WHEELS_OVERLOAD_UNARY_FUNC(sinh)
    WHEELS_OVERLOAD_UNARY_FUNC(asin)
    WHEELS_OVERLOAD_UNARY_FUNC(asinh)
    WHEELS_OVERLOAD_UNARY_FUNC(cos)
    WHEELS_OVERLOAD_UNARY_FUNC(cosh)
    WHEELS_OVERLOAD_UNARY_FUNC(acos)
    WHEELS_OVERLOAD_UNARY_FUNC(acosh)
    WHEELS_OVERLOAD_UNARY_FUNC(tan)
    WHEELS_OVERLOAD_UNARY_FUNC(tanh)
    WHEELS_OVERLOAD_UNARY_FUNC(atan)
    WHEELS_OVERLOAD_UNARY_FUNC(atanh)
    WHEELS_OVERLOAD_UNARY_FUNC(log)
    WHEELS_OVERLOAD_UNARY_FUNC(log2)
    WHEELS_OVERLOAD_UNARY_FUNC(log10)
    WHEELS_OVERLOAD_UNARY_FUNC(exp)
    WHEELS_OVERLOAD_UNARY_FUNC(exp2)
    WHEELS_OVERLOAD_UNARY_FUNC(ceil)
    WHEELS_OVERLOAD_UNARY_FUNC(floor)
    WHEELS_OVERLOAD_UNARY_FUNC(round)
    WHEELS_OVERLOAD_UNARY_FUNC(isinf)
    WHEELS_OVERLOAD_UNARY_FUNC(isfinite)
    WHEELS_OVERLOAD_UNARY_FUNC(isnan)


#define WHEELS_OVERLOAD_BINARY_FUNC(name) \
    struct binary_func_##name { \
        constexpr binary_func_##name() {} \
        template <class TT1, class TT2> \
        constexpr decltype(auto) operator()(TT1 && v1, TT2 && v2) const {\
            using std::name; \
            return name(forward<TT1>(v1), forward<TT2>(v2)); \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class T1, class T2, class = std::enable_if_t< \
        join_overloading<std::decay_t<T1>>::value || \
        join_overloading<std::decay_t<T2>>::value>> \
    constexpr decltype(auto) name(T1 && v1, T2 && v2) { \
        return overloaded<binary_func_##name, \
            info_for_overloading_t<std::decay_t<T1>>, \
            info_for_overloading_t<std::decay_t<T2>> \
        >()(forward<T1>(v1), forward<T2>(v2)); \
    }

    WHEELS_OVERLOAD_BINARY_FUNC(atan2)
    WHEELS_OVERLOAD_BINARY_FUNC(pow)

    
}