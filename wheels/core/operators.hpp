#pragma once

#include <cmath>

#include "utility.hpp"

namespace wheels {

    template <class T>
    struct constant {
        using value_type = T;
        T val;
        template <class TT>
        constexpr constant(TT && v) : val(forward<TT>(v)){}
        template <class ... ArgTs>
        constexpr const T & operator()(ArgTs && ...) const { return val; }
        template <class Archive> void serialize(Archive & ar) { ar(val); }
    };

#define WHEELS_DEFINE_UNARY_OP(op, name) \
        struct unary_op_##name { \
            constexpr unary_op_##name () {} \
            template <class T> \
            constexpr auto operator()(T && v) const {\
                return op forward<T>(v); \
            } \
            template <class Archive> void serialize(Archive &) {} \
        };

    WHEELS_DEFINE_UNARY_OP(+, plus)
    WHEELS_DEFINE_UNARY_OP(-, minus)
    WHEELS_DEFINE_UNARY_OP(!, not)
    WHEELS_DEFINE_UNARY_OP(~, bitwise_not)


#define WHEELS_DEFINE_BINARY_OP(op, name) \
        struct binary_op_##name { \
            constexpr binary_op_##name () {} \
            template <class T1, class T2> \
            constexpr auto operator()(T1 && v1, T2 && v2) const { \
                return forward<T1>(v1) op forward<T2>(v2); \
            } \
            template <class Archive> void serialize(Archive &) {} \
        };

    WHEELS_DEFINE_BINARY_OP(+, plus)
    WHEELS_DEFINE_BINARY_OP(-, minus)
    WHEELS_DEFINE_BINARY_OP(*, mul)
    WHEELS_DEFINE_BINARY_OP(/ , div)
    WHEELS_DEFINE_BINARY_OP(%, mod)
    
    WHEELS_DEFINE_BINARY_OP(== , eq)
    WHEELS_DEFINE_BINARY_OP(!= , neq)
    WHEELS_DEFINE_BINARY_OP(<, lt)
    WHEELS_DEFINE_BINARY_OP(<= , lte)
    WHEELS_DEFINE_BINARY_OP(>, gt)
    WHEELS_DEFINE_BINARY_OP(>= , gte)

    WHEELS_DEFINE_BINARY_OP(&&, and)
    WHEELS_DEFINE_BINARY_OP(|| , or )
    WHEELS_DEFINE_BINARY_OP(&, bitwise_and)
    WHEELS_DEFINE_BINARY_OP(| , bitwise_or)



#define WHEELS_DEFINE_UNARY_OP_WITH(op, name) \
    template <class T> \
    struct unary_op_##name##_with { \
        T val; \
        template <class TT> \
        constexpr unary_op_##name##_with(TT && v) : val(forward<TT>(v)) {} \
        template <class TT> \
        constexpr auto operator()(TT && re) const { \
            return forward<TT>(re) op val; \
        } \
        template <class Archive> void serialize(Archive & ar) {ar(val);} \
    }; \
    template <class T> \
    struct unary_op_##name##_with_ { \
        T val; \
        template <class TT> \
        constexpr unary_op_##name##_with_(TT && v) : val(forward<TT>(v)) {} \
        template <class TT> \
        constexpr auto operator()(TT && re) const { \
            return val op forward<TT>(re); \
        } \
        template <class Archive> void serialize(Archive & ar) {ar(val);} \
    };

    WHEELS_DEFINE_UNARY_OP_WITH(+, plus)
    WHEELS_DEFINE_UNARY_OP_WITH(-, minus)
    WHEELS_DEFINE_UNARY_OP_WITH(*, mul)
    WHEELS_DEFINE_UNARY_OP_WITH(/ , div)
    WHEELS_DEFINE_UNARY_OP_WITH(%, mod)


#define WHEELS_DEFINE_UNARY_FUNC(func) \
    struct unary_func_##func { \
        constexpr unary_func_##func() {} \
        template <class T> \
        constexpr auto operator()(T && t) const { \
            return std::func(forward<T>(t)); \
        } \
        template <class Archive> void serialize(Archive &) {} \
    };

    WHEELS_DEFINE_UNARY_FUNC(sin)
    WHEELS_DEFINE_UNARY_FUNC(cos)
    WHEELS_DEFINE_UNARY_FUNC(tan)
    WHEELS_DEFINE_UNARY_FUNC(sinh)
    WHEELS_DEFINE_UNARY_FUNC(cosh)
    WHEELS_DEFINE_UNARY_FUNC(tanh)
    WHEELS_DEFINE_UNARY_FUNC(asin)
    WHEELS_DEFINE_UNARY_FUNC(acos)
    WHEELS_DEFINE_UNARY_FUNC(atan)
    WHEELS_DEFINE_UNARY_FUNC(asinh)
    WHEELS_DEFINE_UNARY_FUNC(acosh)
    WHEELS_DEFINE_UNARY_FUNC(atanh)
    WHEELS_DEFINE_UNARY_FUNC(exp)
    WHEELS_DEFINE_UNARY_FUNC(exp2)
    WHEELS_DEFINE_UNARY_FUNC(log)
    WHEELS_DEFINE_UNARY_FUNC(log10)
    WHEELS_DEFINE_UNARY_FUNC(log2)
    WHEELS_DEFINE_UNARY_FUNC(abs)
    WHEELS_DEFINE_UNARY_FUNC(sqrt)
    WHEELS_DEFINE_UNARY_FUNC(cbrt)
    WHEELS_DEFINE_UNARY_FUNC(ceil)
    WHEELS_DEFINE_UNARY_FUNC(floor)
    WHEELS_DEFINE_UNARY_FUNC(round)


#define WHEELS_DEFINE_BINARY_FUNC(func) \
    struct binary_func_##func {\
        constexpr binary_func_##func() {} \
        template <class A, class B> \
        constexpr auto operator()(A && a, B && b) const {\
            return std::func(forward<A>(a), forward<B>(b)); \
        } \
        template <class Archive> void serialize(Archive &) {} \
    };
    
    WHEELS_DEFINE_BINARY_FUNC(max)
    WHEELS_DEFINE_BINARY_FUNC(min)
    WHEELS_DEFINE_BINARY_FUNC(pow)
    WHEELS_DEFINE_BINARY_FUNC(atan2)



}