/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include "what_fwd.hpp"

namespace wheels {
// const_expr_base
template <class T> struct const_expr_base;

// invoke_const_expr
template <class E, class... ArgTs>
decltype(auto) invoke_const_expr(E &&e, ArgTs &&... args);

// const_arg
template <size_t Idx> struct const_arg;

namespace literals {
// ""_arg
template <char... Cs> constexpr auto operator"" _arg();
}

template <class T> struct const_coeff;
template <class T> constexpr decltype(auto) as_const_coeff(T &&v);

template <class FunT, class... RecordedArgTs> struct const_call_list;
template <class FunT, class... RecordedExprArgTs>
constexpr auto make_const_call_list(FunT &&f, RecordedExprArgTs &&... as);

template <class OpT> struct func_base;
template <class OpT, class T>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const const_expr_base<T2> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const proxy_base<T2> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const proxy_base<T1> &,
                           const const_expr_base<T2> &);

// has_const_expr
template <class... ArgTs> constexpr auto has_const_expr(const ArgTs &... args);

// smart_invoke
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT &&fun, ArgTs &&... args);
}
