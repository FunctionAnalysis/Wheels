#pragma once

#include <cstdint>
#include <iostream>
#include <array>
#include <tuple>

#include "macros.hpp"
#include "utility.hpp"

namespace wheels {

    namespace details {
        // reduction helper
        template <class T, T ... Vals> struct _reduction {};
        template <class T>
        struct _reduction<T> {
            static constexpr T sum = 0;
            static constexpr T prod = 1;
            static constexpr bool all = true;
            static constexpr bool any = true;
        };
        template <class T, T Val, T ... Vals>
        struct _reduction<T, Val, Vals ...> {
            using _rest_reduction_t = _reduction<T, Vals ...>;
            static constexpr T sum = (T)(Val + _rest_reduction_t::sum);
            static constexpr T prod = (T)(Val * _rest_reduction_t::prod);
            static constexpr bool all = Val && _rest_reduction_t::all;
            static constexpr bool any = Val || _rest_reduction_t::any;
        };
        // element helper
        template <safe_size_t Idx, class T, T ... Vals> struct _element {};
        template <class T, T Val, T ... Vals>
        struct _element<0, T, Val, Vals ...> {
            static constexpr T value = Val;
        };
        template <safe_size_t Idx, class T, T Val, T ... Vals>
        struct _element<Idx, T, Val, Vals ...> {
            static constexpr T value = _element<Idx - 1, T, Vals ...>::value;
        };
    }


    // const_ints
    template <class T, T ... Vals>
    struct const_ints {
        using type = T;
        static constexpr size_t length = sizeof...(Vals);

        constexpr const_ints() {}
        constexpr const_ints() restrict(amp) {}

        static constexpr auto to_array() { return std::array<T, length>{ Vals ... }; }
        static constexpr auto to_tuple() { return std::make_tuple(Vals...); }
        
        static constexpr auto sum() { return const_ints<T, details::_reduction<T, Vals...>::sum>(); }
        static constexpr auto prod() { return const_ints<T, details::_reduction<T, Vals...>::prod>(); }
        static constexpr auto all() { return const_ints<bool, details::_reduction<T, Vals...>::all>(); }
        static constexpr auto any() { return const_ints<bool, details::_reduction<T, Vals...>::any>(); }

        static constexpr auto sum() restrict(amp) { return const_ints<T, details::_reduction<T, Vals...>::sum>(); }
        static constexpr auto prod() restrict(amp) { return const_ints<T, details::_reduction<T, Vals...>::prod>(); }
        static constexpr auto all() restrict(amp) { return const_ints<bool, details::_reduction<T, Vals...>::all>(); }
        static constexpr auto any() restrict(amp) { return const_ints<bool, details::_reduction<T, Vals...>::any>(); }

        template <class K, K Idx>
        constexpr auto operator[](const const_ints<K, Idx> &) const {
            return const_ints<T, details::_element<Idx, T, Vals ...>::value>();
        }
        template <class K, K Idx>
        constexpr auto operator[](const const_ints<K, Idx> &) const restrict(amp) {
            return const_ints<T, details::_element<Idx, T, Vals ...>::value>();
        }

        template <class Archive> void serialize(Archive &) {}
    };


    
    // single value
    template <class T, T Val> 
    struct const_ints<T, Val> {
        using type = T;
        static constexpr size_t length = 1;
        static constexpr T value = Val;

        constexpr const_ints() {}
        constexpr const_ints() restrict(amp) {}
        
        static constexpr auto to_array() { return std::array<T, length>{ Val }; }
        static constexpr auto to_tuple() { return std::make_tuple(Val); }
        
        static constexpr auto sum() { return const_ints<T, Val>(); }
        static constexpr auto prod() { return const_ints<T, Val>(); }
        static constexpr auto all() { return const_ints<bool, (bool)Val>(); }
        static constexpr auto any() { return const_ints<bool, (bool)Val>(); }

        static constexpr auto sum() restrict(amp) { return const_ints<T, Val>(); }
        static constexpr auto prod() restrict(amp) { return const_ints<T, Val>(); }
        static constexpr auto all() restrict(amp) { return const_ints<bool, (bool)Val>(); }
        static constexpr auto any() restrict(amp) { return const_ints<bool, (bool)Val>(); }

        template <class K>
        constexpr auto operator[](const const_ints<K, 0> &) const { return const_ints<T, Val>(); }
        template <class K>
        constexpr auto operator[](const const_ints<K, 0> &) const restrict(amp) { return const_ints<T, Val>(); }

        constexpr operator T() const { return value; }
        template <wheels_enable_if(is_int_supported_by_amp<T>::value)>
        constexpr operator T() const restrict(amp) { return value; }

        template <class Archive> void serialize(Archive &) {}
    };


    template <bool Val> using const_bool = const_ints<bool, Val>;
    template <int Val> using const_int = const_ints<int, Val>;
    template <safe_size_t Val> using const_size = const_ints<safe_size_t, Val>;
    template <safe_size_t Val> using const_index = const_ints<safe_size_t, Val>;

    using yes = const_bool<true>;
    using no = const_bool<false>;





    // is_const_ints
    template <class T>
    struct is_const_ints : no {};
    template <class T, T ... Vals>
    struct is_const_ints<const_ints<T, Vals ...>> : yes {};

    // is_const_int
    template <class T>
    struct is_const_int : no {};
    template <class T, T Val>
    struct is_const_int<const_ints<T, Val>> : yes {};

    // is_int
    template <class T>
    struct is_int : const_bool<(std::is_integral<T>::value || is_const_int<T>::value)> {};



    // conversion with std::integer_sequence
    template <class T, T ... Vals>
    constexpr auto to_const_ints(const std::integer_sequence<T, Vals ...> &) {
        return const_ints<T, Vals ...>();
    }
    template <class T, T ... Vals>
    constexpr auto to_integer_sequence(const const_ints<T, Vals ...> &) {
        return std::integer_sequence<T, Vals ...>();
    }


    // conversion with std::integral_constant
    template <class T, T ... Vals>
    constexpr auto to_const_ints(const std::integral_constant<T, Vals> & ...) {
        return const_ints<T, Vals ...>();
    }
    template <class T, T Val>
    constexpr auto to_integral_constant(const const_ints<T, Val> &) {
        return std::integral_constant<T, Val>();
    }


    // stream
    template <class T, T ... Vals>
    inline std::ostream & operator << (std::ostream & os, const const_ints<T, Vals...> &) {
        return print(" ", os, Vals ...);
    }


    namespace details {
        template <class T, char ... Cs> struct _parse_int {};
        template <class T, char C>
        struct _parse_int<T, C> {
            static_assert(C >= '0' && C <= '9', "invalid character");
            enum : T {
                value = C - '0',
                magnitude = (T)1
            };
        };
        template <class T, char C, char ... Cs>
        struct _parse_int<T, C, Cs...> {
            static_assert(C >= '0' && C <= '9', "invalid character");
            enum : T {
                magnitude = _parse_int<T, Cs...>::magnitude * (T)10,
                rest_value = _parse_int<T, Cs...>::value,
                value = magnitude * (C - '0') + rest_value
            };
        };
    }

    namespace literals {       
        // ""_c
        template <char ... Cs>
        constexpr auto operator "" _c() {
            return const_ints<int, details::_parse_int<int, Cs...>::value>();
        }

        // ""_uc
        template <char ... Cs>
        constexpr auto operator "" _uc() {
            return const_ints<unsigned int, details::_parse_int<unsigned int, Cs...>::value>();
        }

        // ""_sizec
        template <char ... Cs>
        constexpr auto operator "" _sizec() {
            return const_ints<size_t, details::_parse_int<size_t, Cs...>::value>();
        }

        constexpr yes true_c;
        constexpr no false_c;
    }


#define WHEELS_CONST_INT_OVERLOAD_UNARY_OP(op) \
    template <class T, T ... Vals> \
    constexpr auto operator op (const const_ints<T, Vals ...> &) { \
        return const_ints<decltype(op std::declval<T>()), (op Vals) ...>(); \
    } \
    template <class T, T ... Vals> \
    constexpr auto operator op (const const_ints<T, Vals ...> &) restrict(amp) { \
        return const_ints<decltype(op std::declval<T>()), (op Vals) ...>(); \
    }

    WHEELS_CONST_INT_OVERLOAD_UNARY_OP(!)
    WHEELS_CONST_INT_OVERLOAD_UNARY_OP(-)
    WHEELS_CONST_INT_OVERLOAD_UNARY_OP(~)


#define WHEELS_CONST_INT_OVERLOAD_BINARY_OP(op) \
    template <class T1, T1 Val1, class T2, T2 Val2> \
    constexpr auto operator op (const const_ints<T1, Val1> &, const const_ints<T2, Val2> &) {\
        return const_ints<decltype(Val1 op Val2), (Val1 op Val2)>();\
    } \
    template <class T1, T1 Val1, class T2, T2 ... Val2s> \
    constexpr auto operator op (const const_ints<T1, Val1> &, const const_ints<T2, Val2s ...> &) {\
        return const_ints<decltype(Val1 op std::declval<T2>()), (Val1 op Val2s) ...>();\
    } \
    template <class T1, T1 ... Val1s, class T2, T2 Val2> \
    constexpr auto operator op (const const_ints<T1, Val1s ...> &, const const_ints<T2, Val2> &) {\
        return const_ints<decltype(std::declval<T1>() op Val2), (Val1s op Val2) ...>();\
    } \
    template <class T1, T1 ... Val1s, class T2, T2 ... Val2s, \
        class = std::enable_if_t<(sizeof...(Val1s) > 1 && sizeof...(Val2s) > 1)>> \
    constexpr auto operator op (const const_ints<T1, Val1s ...> &, const const_ints<T2, Val2s ...> &) { \
        static_assert(sizeof...(Val1s) == sizeof...(Val2s), "lengths of the two const_ints do not match"); \
        return const_ints<decltype(std::declval<T1>() op std::declval<T2>()), (Val1s op Val2s) ...>();\
    } \
    \
    template <class T1, T1 Val1, class T2, T2 Val2> \
    constexpr auto operator op (const const_ints<T1, Val1> &, const const_ints<T2, Val2> &) restrict(amp) {\
        return const_ints<decltype(Val1 op Val2), (Val1 op Val2)>();\
    } \
    template <class T1, T1 Val1, class T2, T2 ... Val2s> \
    constexpr auto operator op (const const_ints<T1, Val1> &, const const_ints<T2, Val2s ...> &) restrict(amp) {\
        return const_ints<decltype(Val1 op std::declval<T2>()), (Val1 op Val2s) ...>();\
    } \
    template <class T1, T1 ... Val1s, class T2, T2 Val2> \
    constexpr auto operator op (const const_ints<T1, Val1s ...> &, const const_ints<T2, Val2> &) restrict(amp) {\
        return const_ints<decltype(std::declval<T1>() op Val2), (Val1s op Val2) ...>();\
    } \
    template <class T1, T1 ... Val1s, class T2, T2 ... Val2s, \
        class = std::enable_if_t<(sizeof...(Val1s) > 1 && sizeof...(Val2s) > 1)>> \
    constexpr auto operator op (const const_ints<T1, Val1s ...> &, const const_ints<T2, Val2s ...> &) restrict(amp) { \
        static_assert(sizeof...(Val1s) == sizeof...(Val2s), "lengths of the two const_ints do not match"); \
        return const_ints<decltype(std::declval<T1>() op std::declval<T2>()), (Val1s op Val2s) ...>();\
    }


    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(==)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(!=)

    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>=)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<=)

    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(&&)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(||)

    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(&)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(|)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(^)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<<)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>>)

    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(+)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(-)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(*)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(/)
    WHEELS_CONST_INT_OVERLOAD_BINARY_OP(%)



    // cat
    template <class T, T ... Vals>
    constexpr auto cat(const const_ints<T, Vals ...> &) {
        return const_ints<T, Vals ...>();
    }
    namespace details {
        template <class T, T ... Val1s, T ... Val2s>
        constexpr auto _cat2(const const_ints<T, Val1s ...> &, const const_ints<T, Val2s ...> &) {
            return const_ints<T, Val1s ..., Val2s ...>();
        }
    }
    template <class T, class ... Ts>
    constexpr auto cat(const T & first, const Ts & ... rest) {
        static_assert(is_const_ints<T>::value, "");
        return details::_cat2(first, cat(rest...));
    }

    template <class T, T ... Vals>
    constexpr auto cat(const const_ints<T, Vals ...> &) restrict(amp) {
        return const_ints<T, Vals ...>();
    }
    namespace details {
        template <class T, T ... Val1s, T ... Val2s>
        constexpr auto _cat2(const const_ints<T, Val1s ...> &, const const_ints<T, Val2s ...> &) restrict(amp) {
            return const_ints<T, Val1s ..., Val2s ...>();
        }
    }
    template <class T, class ... Ts>
    constexpr auto cat(const T & first, const Ts & ... rest) restrict(amp) {
        static_assert(is_const_ints<T>::value, "");
        return details::_cat2(first, cat(rest...));
    }



    // conditional
    template <class T, T Val, class ThenT, class ElseT>
    constexpr std::enable_if_t<Val, ThenT &&> conditional(const const_ints<T, Val> &,
        ThenT && thenv, ElseT && elsev) {
        return static_cast<ThenT &&>(thenv);
    }
    template <class T, T Val, class ThenT, class ElseT, bool _B = Val>
    constexpr std::enable_if_t<!_B, ElseT &&> conditional(const const_ints<T, Val> &,
        ThenT && thenv, ElseT && elsev) {
        return static_cast<ElseT &&>(elsev);
    }

    template <class T, T Val, class ThenT, class ElseT>
    constexpr std::enable_if_t<Val, ThenT &&> conditional(const const_ints<T, Val> &,
        ThenT && thenv, ElseT && elsev) restrict(amp) {
        return static_cast<ThenT &&>(thenv);
    }
    template <class T, T Val, class ThenT, class ElseT, bool _B = Val>
    constexpr std::enable_if_t<!_B, ElseT &&> conditional(const const_ints<T, Val> &,
        ThenT && thenv, ElseT && elsev) restrict(amp) {
        return static_cast<ElseT &&>(elsev);
    }





    namespace details {
        template <class T, bool IsZero, T N, T ...S> 
        struct _make_seq : _make_seq<T, (N-1==0), N - 1, N - 1, S...> {};
        template <class T, T V, T ...S>
        struct _make_seq<T, true, V, S...> {
            using type = const_ints<T, S...>;
        };

        template <class T, T From, T To, T ...S>
        struct _make_seq_range : _make_seq_range<T, From, To - 1, To - 1, S...> {};
        template <class T, T From, T ...S>
        struct _make_seq_range<T, From, From, S...> {
            using type = const_ints<T, S...>;
        };
    }

    // make_const_sequence
    template <class T, T Size>
    constexpr auto make_const_sequence(const const_ints<T, Size> &) {
        return typename details::_make_seq<T, Size == 0, Size>::type();
    }
    template <class T, T Size>
    constexpr auto make_const_sequence(const const_ints<T, Size> &) restrict(amp) {
        return typename details::_make_seq<T, Size == 0, Size>::type();
    }


    // make_const_range
    template <class T, T From, T To>
    constexpr auto make_const_range(const const_ints<T, From> & from, const const_ints<T, To> & to) {
        return typename details::_make_seq_range<T, From, To>::type();
    }
    template <class T, T From, T To>
    constexpr auto make_const_range(const const_ints<T, From> & from, const const_ints<T, To> & to) restrict(amp) {
        return typename details::_make_seq_range<T, From, To>::type();
    }

}