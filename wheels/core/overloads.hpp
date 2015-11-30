#pragma once

#include "constants.hpp"

namespace wheels {

    template <class T>
    struct join_overloading : no {};

    template <class T>
    struct info_for_overloading {
        using type = T;
    };
    template <class T>
    using info_for_overloading_t = typename info_for_overloading<T>::type;



#define WHEELS_OVERLOAD_UNARY_OP(op, name) \
    template <class InfoT> \
    struct overloaded_unary_op_##name { \
        constexpr overloaded_unary_op_##name() {} \
        template <class TT> \
        constexpr auto operator()(TT && v) const { \
            static_assert(always<bool, false, TT>::value, "not implemented yet"); \
        } \
        template <class Archive> void serialize(Archive & ar) {} \
    }; \
    template <class T, \
        class = std::enable_if_t<join_overloading<std::decay_t<T>>::value>> \
    constexpr auto operator op (T && v) { \
        return overloaded_unary_op_##name<info_for_overloading_t<std::decay_t<T>>>()(forward<T>(v)); \
    }

    WHEELS_OVERLOAD_UNARY_OP(-, minus)
    WHEELS_OVERLOAD_UNARY_OP(!, not)
    WHEELS_OVERLOAD_UNARY_OP(~, bitwise_not)


#define WHEELS_OVERLOAD_BINARY_OP(op, name) \
    template <class InfoT1, class InfoT2> \
    struct overloaded_binary_op_##name { \
        constexpr overloaded_binary_op_##name() {} \
        template <class TT1, class TT2> \
        constexpr auto operator()(TT1 &&, TT2 &&) const { \
            static_assert(always<bool, false, TT1, TT2>::value, "not implemented yet"); \
        } \
        template <class Archive> void serialize(Archive & ar) {} \
    }; \
    template <class T1, class T2, class = std::enable_if_t< \
        join_overloading<std::decay_t<T1>>::value || \
        join_overloading<std::decay_t<T2>>::value>> \
    constexpr auto operator op (T1 && v1, T2 && v2) { \
        return overloaded_binary_op_##name< \
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







}