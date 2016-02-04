#pragma once

#include <cassert>

#include "../core/const_expr.hpp"
#include "../core/constants.hpp"
#include "../core/overloads.hpp"
#include "../core/parallel.hpp"
#include "../core/serialize.hpp"
#include "../core/types.hpp"

#include "shape.hpp"

namespace wheels {

template <class ShapeT, class EleT, class T> struct category_tensor {};

namespace index_tags {
constexpr auto first = const_index<0>();
constexpr auto length = const_symbol<0>();
constexpr auto last = length - const_index<1>();
}

namespace details {
template <class E, class SizeT, class = std::enable_if_t<!is_int<E>::value>>
constexpr auto _eval_index_expr(const E &e, const SizeT &sz) {
  return e(sz);
}
template <class T, class SizeT, class = std::enable_if_t<is_int<T>::value>,
          class = void>
constexpr auto _eval_index_expr(const T &t, const SizeT &) {
  return t;
}
}

// inheritance of tensor class T:
// tensor_core<T> -> tensor_base<ShapeT, ET, T> [->
// tensor_op_result_base<ShapeT, ET, OpT, T>] -> T

// tensor_core
template <class T> struct tensor_core {
  const tensor_core &core() const { return *this; }
  constexpr const T &derived() const { return static_cast<const T &>(*this); }
  T &derived() { return static_cast<T &>(*this); }

  constexpr auto shape() const { return ::wheels::shape_of(derived()); }
  template <class K, K Idx>
  constexpr auto size(const const_ints<K, Idx> &i) const {
    return ::wheels::size_at(derived(), i);
  }
  constexpr auto numel() const { return ::wheels::numel(derived()); }

  constexpr auto norm_squared() const {
    return ::wheels::norm_squared(derived());
  }
  constexpr auto norm() const { return ::wheels::norm(derived()); }
  constexpr auto normalized() const & { return derived() / this->norm(); }
  auto normalized() && { return std::move(derived()) / this->norm(); }

  constexpr auto sum() const { return ::wheels::sum_of(derived()); }

  // at_or(otherwisev, subs ...)
  template <class E, class... SubTs>
  constexpr decltype(auto) at_or(E &&otherwise, const SubTs &... subs) const {
    return _at_or_seq(forward<E>(otherwise),
                      make_const_sequence_for<SubTs...>(), subs...);
  }

  // operator()(subs ...)
  template <class... SubTs>
  constexpr decltype(auto) operator()(const SubTs &... subs) const {
    return _parenthesis_seq(make_const_sequence_for<SubTs...>(), subs...);
  }
  template <class... SubTs> decltype(auto) operator()(const SubTs &... subs) {
    return _parenthesis_seq(make_const_sequence_for<SubTs...>(), subs...);
  }

  // operator[](index)
  template <class E> constexpr decltype(auto) operator[](const E &e) const {
    return ::wheels::element_at_index(derived(),
                                      details::_eval_index_expr(e, numel()));
  }
  template <class E> decltype(auto) operator[](const E &e) {
    return ::wheels::element_at_index(derived(),
                                      details::_eval_index_expr(e, numel()));
  }

  // for_each
  template <class FunT> void for_each(FunT &fun) const {
    ::wheels::for_each_element(behavior_flag<unordered>(), fun, derived());
  }
  template <class FunT> void for_each(FunT &fun) {
    ::wheels::for_each_element(behavior_flag<unordered>(), fun, derived());
  }

  // transform
  template <class FunT> auto transform(FunT &&fun) const & {
    return ::wheels::transform(derived(), forward<FunT>(fun));
  }
  template <class FunT> auto transform(FunT &&fun) && {
    return ::wheels::transform(std::move(derived()), forward<FunT>(fun));
  }

private:
  template <class E, class... SubEs, size_t... Is>
  constexpr decltype(auto) _at_or_seq(E &&otherwise,
                                      const_ints<size_t, Is...> seq,
                                      const SubEs &... subes) const {
    return ::wheels::all(::wheels::is_between(
               details::_eval_index_expr(subes, size(const_size<Is>())), 0,
               size(const_size<Is>()))...)
               ? _parenthesis_seq(seq, subes...)
               : forward<E>(otherwise);
  }

  template <class... SubEs, size_t... Is>
  constexpr decltype(auto) _parenthesis_seq(const_ints<size_t, Is...>,
                                            const SubEs &... subes) const {
    return ::wheels::element_at(
        derived(), details::_eval_index_expr(subes, size(const_size<Is>()))...);
  }
  template <class... SubEs, size_t... Is>
  decltype(auto) _parenthesis_seq(const_ints<size_t, Is...>,
                                  const SubEs &... subes) {
    return ::wheels::element_at(
        derived(), details::_eval_index_expr(subes, size(const_size<Is>()))...);
  }
};

template <class ShapeT, class ET> class tensor;

// tensor_base<ShapeT, ET, T>
template <class ShapeT, class ET, class T> struct tensor_base : tensor_core<T> {
  using shape_type = ShapeT;
  static constexpr size_t rank = ShapeT::rank;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<ShapeT, ET> eval() const {
    return tensor<ShapeT, ET>(derived());
  }
  constexpr operator tensor<ShapeT, ET>() const { return eval(); }
};

// 1 dimensional tensor (vector)
template <class ST, class NT, class ET, class T>
struct tensor_base<tensor_shape<ST, NT>, ET, T> : tensor_core<T> {
  using shape_type = tensor_shape<ST, NT>;
  static constexpr size_t rank = 1;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<shape_type, value_type> eval() const {
    return tensor<shape_type, value_type>(derived());
  }
  constexpr operator tensor<shape_type, value_type>() const { return eval(); }

  // xyzw
  constexpr decltype(auto) x() const {
    return ::wheels::element_at(derived(), 0);
  }
  constexpr decltype(auto) y() const {
    return ::wheels::element_at(derived(), 1);
  }
  constexpr decltype(auto) z() const {
    return ::wheels::element_at(derived(), 2);
  }
  constexpr decltype(auto) w() const {
    return ::wheels::element_at(derived(), 3);
  }

  decltype(auto) x() { return ::wheels::element_at(derived(), 0); }
  decltype(auto) y() { return ::wheels::element_at(derived(), 1); }
  decltype(auto) z() { return ::wheels::element_at(derived(), 2); }
  decltype(auto) w() { return ::wheels::element_at(derived(), 3); }

  // rgba
  constexpr decltype(auto) r() const {
    return ::wheels::element_at(derived(), 0);
  }
  constexpr decltype(auto) g() const {
    return ::wheels::element_at(derived(), 1);
  }
  constexpr decltype(auto) b() const {
    return ::wheels::element_at(derived(), 2);
  }
  constexpr decltype(auto) a() const {
    return ::wheels::element_at(derived(), 3);
  }

  decltype(auto) r() { return ::wheels::element_at(derived(), 0); }
  decltype(auto) g() { return ::wheels::element_at(derived(), 1); }
  decltype(auto) b() { return ::wheels::element_at(derived(), 2); }
  decltype(auto) a() { return ::wheels::element_at(derived(), 3); }

  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  dot(const tensor_base<tensor_shape<ST2, NT2>, ET2, T2> &t) const {
    return ::wheels::dot(*this, t);
  }
  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  cross(const tensor_base<tensor_shape<ST2, NT2>, ET2, T2> &t) const {
    return ::wheels::cross(*this, t);
  }
};

// 2 dimensional tensor (matrix)
template <class ST, class MT, class NT, class ET, class T>
struct tensor_base<tensor_shape<ST, MT, NT>, ET, T> : tensor_core<T> {
  using shape_type = tensor_shape<ST, MT, NT>;
  static constexpr size_t rank = 2;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<shape_type, value_type> eval() const {
    return tensor<shape_type, value_type>(derived());
  }
  constexpr operator tensor<shape_type, value_type>() const { return eval(); }

  constexpr auto rows() const { return size(const_index<0>()); }
  constexpr auto cols() const { return size(const_index<1>()); }

  constexpr auto t() const & { return ::wheels::transpose(derived()); }
  auto t() & { return ::wheels::transpose(derived()); }
  auto t() && { return ::wheels::transpose(std::move(derived())); }
};

// 3 dimensional (only third dimension is static) tensor (image)
template <class ST, ST D, class ET, class T>
struct tensor_base<tensor_shape<ST, ST, ST, const_ints<ST, D>>, ET, T>
    : tensor_core<T> {
  using shape_type = tensor_shape<ST, ST, ST, const_ints<ST, D>>;
  static constexpr size_t rank = 3;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<shape_type, value_type> eval() const {
    return tensor<shape_type, value_type>(derived());
  }
  constexpr operator tensor<shape_type, value_type>() const { return eval(); }

  constexpr auto rows() const { return size(const_index<0>()); }
  constexpr auto cols() const { return size(const_index<1>()); }
  constexpr auto pixels() const { return rows() * cols(); }
  static constexpr auto channels() { return const_ints<ST, D>(); }
};

// category_for_overloading
// common_func
template <class ShapeT, class ET, class T, class OpT>
constexpr auto category_for_overloading(const tensor_base<ShapeT, ET, T> &,
                                        const common_func<OpT> &) {
  return category_tensor<ShapeT, ET, T>();
}

// tensor_op_result_base
template <class ShapeT, class EleT, class OpT, class T>
struct tensor_op_result_base : tensor_base<ShapeT, EleT, T> {};

// t1 == t2
template <class ShapeT, class T>
struct tensor_op_result_base<ShapeT, bool, binary_op_eq, T>
    : tensor_base<ShapeT, bool, T> {
  constexpr operator bool() const {
    return ::wheels::equals_result_of(derived());
  }
};

// t1 != t2
template <class ShapeT, class T>
struct tensor_op_result_base<ShapeT, bool, binary_op_neq, T>
    : tensor_base<ShapeT, bool, T> {
  constexpr operator bool() const {
    return ::wheels::not_equals_result_of(derived());
  }
};

// -- necessary tensor functions
// Shape shape_of(ts);
template <class T>
constexpr tensor_shape<size_t> shape_of(const tensor_core<T> &) {
  static_assert(always<bool, false, T>::value,
                "shape_of(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
}

// Scalar element_at(ts, subs ...);
template <class T, class... SubTs>
constexpr double element_at(const tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
}
template <class T, class... SubTs>
constexpr double &element_at(tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(T &) is not supported by tensor_core<T>, do you "
                "forget to call .derived()?");
}

// -- auxiliary tensor functions
// auto size_at(ts, const_int);
template <class T, class ST, ST Idx>
constexpr auto size_at(const tensor_core<T> &t,
                       const const_ints<ST, Idx> &idx) {
  return shape_of(t.derived()).at(idx);
}

// auto numel(ts)
template <class T> constexpr auto numel(const tensor_core<T> &t) {
  return shape_of(t.derived()).magnitude();
}

// Scalar element_at_index(ts, index);
template <class T, class IndexT>
constexpr decltype(auto) element_at_index(const tensor_core<T> &t,
                                          const IndexT &ind) {
  return invoke_with_subs(shape_of(t.derived()), ind, [&t](auto &&... subs) {
    return element_at(t.derived(), subs...);
  });
}

// void reserve_shape(ts, shape);
template <class T, class ST, class... SizeTs>
void reserve_shape(tensor_core<T> &, const tensor_shape<ST, SizeTs...> &shape) {
}

// behavior_flag used in for_each_element*
enum behavior_flag_enum {
  index_ascending,
  unordered,
  break_on_false,
  nonzero_only
};
template <behavior_flag_enum O>
using behavior_flag = const_ints<behavior_flag_enum, O>;

// index_ascending
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      const tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for_each_subscript(t.shape(), [&fun, &t, &ts...](auto &... subs) {
    fun(element_at(t.derived(), subs...), element_at(ts.derived(), subs...)...);
  });
  return true;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for_each_subscript(t.shape(), [&fun, &t, &ts...](auto &... subs) {
    fun(element_at(t.derived(), subs...), element_at(ts.derived(), subs...)...);
  });
  return true;
}

// unordered
namespace details {
template <class FunT, class T, class... Ts>
void _for_each_element_unordered_default(yes staticShape, FunT &fun, T &t,
                                         Ts &... ts) {
  for_each_element(behavior_flag<index_ascending>(), fun, t, ts...);
}
constexpr size_t _numel_parallel_thres = (size_t)4e4;
template <class FunT, class T, class... Ts>
void _for_each_element_unordered_default(no staticShape, FunT &fun, T &t,
                                         Ts &... ts) {
  if (t.numel() < _numel_parallel_thres) {
    for_each_element(behavior_flag<index_ascending>(), fun, t, ts...);
  } else {
    parallel_for_each(t.numel(),
                      [&](size_t i) {
                        fun(element_at_index(t, i), element_at_index(ts, i)...);
                      },
                      _numel_parallel_thres / 2);
  }
}
}
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT &fun,
                      const tensor_core<T> &t, Ts &... ts) {
  details::_for_each_element_unordered_default(
      const_bool<::wheels::any(
          std::decay_t<decltype(t.shape())>::is_static,
          std::decay_t<decltype(ts.shape())>::is_static...)>(),
      fun, t.derived(), ts.derived()...);
  return true;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT &fun, tensor_core<T> &t,
                      Ts &... ts) {
  details::_for_each_element_unordered_default(
      const_bool<::wheels::any(
          std::decay_t<decltype(t.shape())>::is_static,
          std::decay_t<decltype(ts.shape())>::is_static...)>(),
      fun, t.derived(), ts.derived()...);
  return true;
}

// break_on_false
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      const tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  return for_each_subscript_if(t.shape(), [&](auto &... subs) {
    return fun(element_at(t.derived(), subs...),
               element_at(ts.derived(), subs...)...);
  });
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  return for_each_subscript_if(t.shape(), [&](auto &... subs) {
    return fun(element_at(t.derived(), subs...),
               element_at(ts.derived(), subs...)...);
  });
}

// nonzero_only
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      const tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  bool visited_all = true;
  for_each_subscript(t.shape(), [&](auto &... subs) {
    decltype(auto) e = element_at(t.derived(), subs...);
    if (e) {
      fun(e, element_at(ts.derived(), subs...)...);
    } else {
      visited_all = false;
    }
  });
  return visited_all;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      tensor_core<T> &t, Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  bool visited_all = true;
  for_each_subscript(t.shape(), [&](auto &... subs) {
    decltype(auto) e = element_at(t.derived(), subs...);
    if (e) {
      fun(e, element_at(ts.derived(), subs...)...);
    } else {
      visited_all = false;
    }
  });
  return visited_all;
}

// void assign_elements(to, from);
template <class ToT, class FromT>
void assign_elements(tensor_core<ToT> &to, const tensor_core<FromT> &from) {
  decltype(auto) s = from.shape();
  if (to.shape() != s) {
    reserve_shape(to.derived(), s);
  }
  for_each_element(behavior_flag<unordered>(),
                   [](auto &to_e, auto from_e) { to_e = from_e; }, to.derived(),
                   from.derived());
}

// Scalar reduce_elements(ts, initial, functor);
template <class T, class E, class ReduceT>
E reduce_elements(const tensor_core<T> &t, E initial, ReduceT &red) {
  for_each_element(behavior_flag<unordered>(),
                   [&initial, &red](auto &e) { initial = red(initial, e); },
                   t.derived());
  return initial;
}

// Scalar norm_squared(ts)
template <class ShapeT, class ET, class T>
ET norm_squared(const tensor_base<ShapeT, ET, T> &t) {
  ET result = 0.0;
  for_each_element(behavior_flag<nonzero_only>(),
                   [&result](auto &&e) { result += e * e; }, t.derived());
  return result;
}

// Scalar norm(ts)
template <class ShapeT, class ET, class T>
constexpr ET norm(const tensor_base<ShapeT, ET, T> &t) {
  return sqrt(norm_squared(t.derived()));
}

// bool all(s)
template <class ShapeT, class ET, class T>
constexpr bool all_of(const tensor_base<ShapeT, ET, T> &t) {
  return for_each_element(behavior_flag<break_on_false>(),
                          [](auto &&e) { return !!e; }, t.derived());
}

// bool any(s)
template <class ShapeT, class ET, class T>
constexpr bool any_of(const tensor_base<ShapeT, ET, T> &t) {
  return !for_each_element(behavior_flag<break_on_false>(),
                           [](auto &&e) { return !e; }, t.derived());
}

// equals_result_of
template <class ShapeT, class ET, class T>
constexpr bool equals_result_of(const tensor_base<ShapeT, ET, T> &t) {
  return all_of(t);
}

// not_equals_result_of
template <class ShapeT, class ET, class T>
constexpr bool not_equals_result_of(const tensor_base<ShapeT, ET, T> &t) {
  return any_of(t);
}

// Scalar sum(s)
template <class ShapeT, class ET, class T>
constexpr ET sum_of(const tensor_base<ShapeT, ET, T> &t) {
  ET s = types<ET>::zero();
  for_each_element(behavior_flag<nonzero_only>(),
                   [&s](const auto &e) { s += e; }, t.derived());
  return s;
}
}
