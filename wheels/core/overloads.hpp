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
                "error: this overloaded operator/function is not implemented, "
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
            return (op forward<TT>(v)); \
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



#define WHEELS_OVERLOAD_FUNC(name) \
    struct func_##name { \
        constexpr func_##name() {} \
        template <class ... ArgTs> \
        constexpr decltype(auto) operator()(ArgTs && ... vs) const {\
            using std::name;\
            return name(forward<ArgTs>(vs) ...); \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class FirstT, class ... RestTs, \
        class = std::enable_if_t<any( \
            join_overloading<std::decay_t<FirstT>>::value, \
            join_overloading<std::decay_t<RestTs>>::value ...)>, \
        class = void> \
    constexpr decltype(auto) name(FirstT && f, RestTs && ... rests) { \
        return overloaded<func_##name, \
            info_for_overloading_t<std::decay_t<FirstT>>, \
            info_for_overloading_t<std::decay_t<RestTs>> ...\
        >()(forward<FirstT>(f), forward<RestTs>(rests) ...); \
    }

    WHEELS_OVERLOAD_FUNC(sin)
    WHEELS_OVERLOAD_FUNC(sinh)
    WHEELS_OVERLOAD_FUNC(asin)
    WHEELS_OVERLOAD_FUNC(asinh)
    WHEELS_OVERLOAD_FUNC(cos)
    WHEELS_OVERLOAD_FUNC(cosh)
    WHEELS_OVERLOAD_FUNC(acos)
    WHEELS_OVERLOAD_FUNC(acosh)
    WHEELS_OVERLOAD_FUNC(tan)
    WHEELS_OVERLOAD_FUNC(tanh)
    WHEELS_OVERLOAD_FUNC(atan)
    WHEELS_OVERLOAD_FUNC(atanh)
    WHEELS_OVERLOAD_FUNC(log)
    WHEELS_OVERLOAD_FUNC(log2)
    WHEELS_OVERLOAD_FUNC(log10)
    WHEELS_OVERLOAD_FUNC(exp)
    WHEELS_OVERLOAD_FUNC(exp2)
    WHEELS_OVERLOAD_FUNC(ceil)
    WHEELS_OVERLOAD_FUNC(floor)
    WHEELS_OVERLOAD_FUNC(round)
    WHEELS_OVERLOAD_FUNC(isinf)
    WHEELS_OVERLOAD_FUNC(isfinite)
    WHEELS_OVERLOAD_FUNC(isnan)
    
    WHEELS_OVERLOAD_FUNC(atan2)
    WHEELS_OVERLOAD_FUNC(pow)



    

    // overload member functions
    template <class DerivedT, class OpT>
    struct object_overloading {};
    template <class DerivedT, class ... OpTs>
    struct object_overloadings : object_overloading<DerivedT, OpTs> ... {};

#define WHEELS_OVERLOAD_MEMBER_UNARY_OP(op1, op2, op3, opsymbol, name) \
    struct member_op_##name { \
        constexpr member_op_##name() {} \
        template <class CallerT, class ArgT> \
        constexpr decltype(auto) operator()(CallerT && caller, ArgT && arg) const { \
            return op1 forward<CallerT>(caller) op2 forward<ArgT>(arg) op3; \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class DerivedT> \
    struct object_overloading<DerivedT, member_op_##name> { \
        template <class ArgT> \
        constexpr decltype(auto) operator opsymbol (ArgT && arg) const & { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgT>> \
            >()(static_cast<const DerivedT &>(*this), forward<ArgT>(arg)); \
        } \
        template <class ArgT> \
        decltype(auto) operator opsymbol (ArgT && arg) & { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgT>> \
            >()(static_cast<DerivedT &>(*this), forward<ArgT>(arg)); \
        } \
        template <class ArgT> \
        decltype(auto) operator opsymbol (ArgT && arg) && { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgT>> \
            >()(static_cast<DerivedT &&>(*this), forward<ArgT>(arg)); \
        } \
        template <class ArgT> \
        decltype(auto) operator opsymbol (ArgT && arg) const && { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgT>> \
            >()(static_cast<const DerivedT &&>(*this), forward<ArgT>(arg)); \
        } \
    };


    WHEELS_OVERLOAD_MEMBER_UNARY_OP( , [, ], [], bracket)
    WHEELS_OVERLOAD_MEMBER_UNARY_OP( , +=, , +=, plus_equal)
    WHEELS_OVERLOAD_MEMBER_UNARY_OP( , -=, , -=, minus_equal)
    WHEELS_OVERLOAD_MEMBER_UNARY_OP( , *=, , *=, mul_equal)
    WHEELS_OVERLOAD_MEMBER_UNARY_OP( , /=, , /=, div_equal)
    WHEELS_OVERLOAD_MEMBER_UNARY_OP( ,  =, ,  =, assign)
    


#define WHEELS_OVERLOAD_MEMBER_VARARG_OP(op1, op2, op3, opsymbol, name) \
    struct member_op_##name { \
        constexpr member_op_##name() {} \
        template <class CallerT, class ... ArgTs> \
        constexpr decltype(auto) operator()(CallerT && caller, ArgTs && ... args) const { \
            return op1 forward<CallerT>(caller) op2 forward<ArgTs>(args) ... op3; \
        } \
        template <class Archive> \
        void serialize(Archive &) {} \
    }; \
    template <class DerivedT> \
    struct object_overloading<DerivedT, member_op_##name> { \
        template <class ... ArgTs> \
        constexpr decltype(auto) operator opsymbol (ArgTs && ... args) const & { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgTs>> ... \
            >()(static_cast<const DerivedT &>(*this), forward<ArgTs>(args) ...); \
        } \
        template <class ... ArgTs> \
        decltype(auto) operator opsymbol (ArgTs && ... args) & { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgTs>> ... \
            >()(static_cast<DerivedT &>(*this), forward<ArgTs>(args) ...); \
        } \
        template <class ... ArgTs> \
        decltype(auto) operator opsymbol (ArgTs && ... args) && { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgTs>> ... \
            >()(static_cast<DerivedT &&>(*this), forward<ArgTs>(args) ...); \
        } \
        template <class ... ArgTs> \
        decltype(auto) operator opsymbol (ArgTs && ... args) const && { \
            return overloaded<member_op_##name, \
                info_for_overloading_t<std::decay_t<DerivedT>>, \
                info_for_overloading_t<std::decay_t<ArgTs>> ... \
            >()(static_cast<const DerivedT &&>(*this), forward<ArgTs>(args) ...); \
        } \
    };


#define WHEELS_SYMBOL_LEFT_PAREN (
#define WHEELS_SYMBOL_RIGHT_PAREN )
    WHEELS_OVERLOAD_MEMBER_VARARG_OP( , WHEELS_SYMBOL_LEFT_PAREN, WHEELS_SYMBOL_RIGHT_PAREN, (), paren)


}