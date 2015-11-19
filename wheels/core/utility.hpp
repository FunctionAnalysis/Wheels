#pragma once

#include <iostream>
#include "macros.hpp"

namespace wheels {

    // if_then_else for enumulating ?:
    template <class ThenT, class ElseT>
    constexpr auto conditional(bool b, ThenT && thenv, ElseT && elsev) {
        return b ? std::forward<ThenT>(thenv) : std::forward<ElseT>(elsev);
    }



    // all(...)
    template <class T>
    constexpr T && all(T && v) { return static_cast<T &&>(v); }
    template <class T, class ... Ts>
    constexpr auto all(T && v, Ts && ... vs) { return std::forward<T>(v) && all(std::forward<Ts>(vs)...); }

    // any(...)
    template <class T>
    constexpr T && any(T && v) { return static_cast<T &&>(v); }
    template <class T, class ... Ts>
    constexpr auto any(T && v, Ts && ... vs) { return std::forward<T>(v) || any(std::forward<Ts>(vs)...); }
    
    // sum(...)
    template <class T>
    constexpr T && sum(T && v) { return static_cast<T &&>(v); }
    template <class T, class ... Ts>
    constexpr auto sum(T && v, Ts && ... vs) { return std::forward<T>(v) + sum(std::forward<Ts>(vs)...); }

    // prod(...)
    template <class T>
    constexpr T && prod(T && v) { return static_cast<T &&>(v); }
    template <class T, class ... Ts>
    constexpr auto prod(T && v, Ts && ... vs) { return std::forward<T>(v) * prod(std::forward<Ts>(vs)...); }

    // min(...)
    template <class T>
    constexpr T && min(T && v) { return static_cast<T &&>(v); }
    namespace details {
        template <class T1, class T2>
        constexpr auto _min2(T1 && a, T2 && b) {
            return conditional(a < b, std::forward<T1>(a), std::forward<T2>(b));
        }
    }
    template <class T, class ... Ts>
    constexpr auto min(T && v, Ts && ... vs) {
        return details::_min2(std::forward<T>(v), min(std::forward<Ts>(vs)...));
    }

    // max(...)
    template <class T>
    constexpr T && max(T && v) { return static_cast<T &&>(v); }
    namespace details {
        template <class T1, class T2>
        constexpr auto _max2(T1 && a, T2 && b) {
            return conditional(a < b, std::forward<T2>(b), std::forward<T1>(a));
        }
    }
    template <class T, class ... Ts>
    constexpr auto max(T && v, Ts && ... vs) {
        return details::_max2(std::forward<T>(v), max(std::forward<Ts>(vs)...));
    }

    

    // traverse(fun, ...)
    template <class FunT, class T>
    constexpr void traverse(const FunT & fun, const T & v) { fun(v); }
    template <class FunT, class T, class ... Ts>
    constexpr void traverse(const FunT & fun, const T & v, const Ts & ... vs) {
        fun(v); traverse(fun, vs ...);
    }



    // always
    template <class T, T Val, class ... ArgTs> 
    struct always {
        static constexpr T value = Val;
    };




    // print
    inline std::ostream & print(std::ostream & os) { return os; }
    template <class T, class ... Ts>
    inline std::ostream & print(std::ostream & os, const T & arg, const Ts & ... args) {
        os << arg;
        return print(os, args ...);
    }
    template <class SepT>
    inline std::ostream & print(SepT && sep, std::ostream & os) { return os; }
    template <class SepT, class T>
    inline std::ostream & print(SepT && sep, std::ostream & os, const T & arg) { return os << arg; }
    template <class SepT, class T, class ... Ts>
    inline std::ostream & print(SepT && sep, std::ostream & os, const T & arg, const Ts & ... args) { 
        os << arg << sep;
        return print(sep, os, args ...);
    }

    // println
    template <class ... Ts>
    inline std::ostream & println(std::ostream & os, const Ts & ... args) {
        return print(os, args...) << std::endl;
    }
    template <class SepT, class ... Ts>
    inline std::ostream & println(SepT && sep, std::ostream & os, const Ts & ... args) {
        return print(std::forward<SepT>(sep), os, args...) << std::endl;
    }
   

}