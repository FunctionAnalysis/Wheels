#pragma once

#include <cassert>

#include "../core/const_expr.hpp"
#include "../core/constants.hpp"
#include "../core/overloads.hpp"
#include "../core/serialize.hpp"
#include "../core/types.hpp"

#include "shape.hpp"

namespace wheels {

template <class ShapeT, class EleT, class T> struct category_tensor {};
//

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

private:
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

// tensor_base_<ET, T>
template <class ET, class T> struct tensor_base_ : tensor_core<T> {
  using value_type = ET;
};
template <class T> struct tensor_base_<bool, T> : tensor_core<T> {
  using value_type = bool;
  constexpr operator bool() const { return ::wheels::all_of(derived()); }
};

// tensor_base<ShapeT, ET, T>
template <class ShapeT, class ET, class T>
struct tensor_base : tensor_base_<ET, T> {
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
struct tensor_base<tensor_shape<ST, NT>, ET, T> : tensor_base_<ET, T> {
  using shape_type = tensor_shape<ST, NT>;
  static constexpr size_t rank = 1;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<tensor_shape<ST, NT>, ET> eval() const {
    return tensor<tensor_shape<ST, NT>, ET>(derived());
  }
  constexpr operator tensor<tensor_shape<ST, NT>, ET>() const { return eval(); }

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
struct tensor_base<tensor_shape<ST, MT, NT>, ET, T> : tensor_base_<ET, T> {
  using shape_type = tensor_shape<ST, MT, NT>;
  static constexpr size_t rank = 2;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  constexpr tensor<tensor_shape<ST, MT, NT>, ET> eval() const {
    return tensor<tensor_shape<ST, MT, NT>, ET>(derived());
  }
  constexpr operator tensor<tensor_shape<ST, MT, NT>, ET>() const {
    return eval();
  }

  constexpr auto rows() const { return size(const_index<0>()); }
  constexpr auto cols() const { return size(const_index<1>()); }

  constexpr auto t() const & { return ::wheels::transpose(derived()); }
  auto t() & { return ::wheels::transpose(derived()); }
  auto t() && { return ::wheels::transpose(std::move(derived())); }
};

// category_for_overloading
// common_func
template <class ShapeT, class ET, class T, class OpT>
constexpr auto category_for_overloading(const tensor_base<ShapeT, ET, T> &,
                                        const common_func<OpT> &) {
  return category_tensor<ShapeT, ET, T>();
}

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

// order_flag used in for_each_element*
enum order_flag_enum { index_ascending, unordered };
template <order_flag_enum O> using order_flag = const_ints<order_flag_enum, O>;

// for_each_element
template <class FunT, class T, class... Ts>
void for_each_element(order_flag<index_ascending>, FunT &&fun, T &&t,
                      Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for_each_subscript(shape_of(t), [&](auto &&... subs) {
    fun(element_at(t, subs...), element_at(ts, subs...)...);
  });
}

template <class FunT, class T, class... Ts>
void for_each_element(order_flag<unordered>, FunT &&fun, T &&t, Ts &&... ts) {
  for_each_element(order_flag<index_ascending>(), forward<FunT>(fun),
                   forward<T>(t), forward<Ts>(ts)...);
}

// for_each
template <class FunT, class T, class... Ts>
void for_each(FunT &&fun, T &&t, Ts &&... ts) {
  for_each_element(order_flag<unordered>(), forward<FunT>(fun), forward<T>(t),
                   forward<Ts>(ts)...);
}

// for_each_element_with_short_circuit
template <class FunT, class T, class... Ts>
bool for_each_element_with_short_circuit(order_flag<index_ascending>,
                                         FunT &&fun, T &&t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  return for_each_subscript_if(shape_of(t), [&](auto &&... subs) {
    return fun(element_at(t, subs...), element_at(ts, subs...)...);
  });
}

template <class FunT, class T, class... Ts>
bool for_each_element_with_short_circuit(order_flag<unordered>, FunT &&fun,
                                         T &&t, Ts &&... ts) {
  return for_each_element_with_short_circuit(order_flag<index_ascending>(),
                                             forward<FunT>(fun), forward<T>(t),
                                             forward<Ts>(ts)...);
}

// for_each_nonzero_element
template <class FunT, class T, class... Ts>
void for_each_nonzero_element(order_flag<index_ascending>, FunT &&fun, T &&t,
                              Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for_each_subscript(shape_of(t), [&](auto &&... subs) {
    decltype(auto) e = element_at(t, subs...);
    if (e) {
      fun(e, element_at(ts, subs...)...);
    }
  });
}

template <class FunT, class T, class... Ts>
void for_each_nonzero_element(order_flag<unordered>, FunT &&fun, T &&t,
                              Ts &&... ts) {
  for_each_nonzero_element(order_flag<index_ascending>(), forward<FunT>(fun),
                           forward<T>(t), forward<Ts>(ts)...);
}

// void assign_elements(to, from);
template <class To, class From>
void assign_elements(tensor_core<To> &to, const tensor_core<From> &from) {
  decltype(auto) s = shape_of(from.derived());
  if (shape_of(to.derived()) != s) {
    reserve_shape(to.derived(), s);
  }
  for_each_element(order_flag<unordered>(),
                   [](auto &to_e, const auto from_e) { to_e = from_e; },
                   to.derived(), from.derived());
}

// Scalar reduce_elements(ts, initial, functor);
template <class T, class E, class ReduceT>
E reduce_elements(const T &t, E initial, ReduceT &&red) {
  for_each_element(order_flag<unordered>(),
                   [&initial, &red](auto &&e) { initial = red(initial, e); },
                   t);
  return initial;
}

// Scalar norm_squared(ts)
template <class ShapeT, class ET, class T>
ET norm_squared(const tensor_base<ShapeT, ET, T> &t) {
  ET result = 0.0;
  for_each_nonzero_element(order_flag<unordered>(),
                           [&result](auto &&e) { result += e * e; },
                           t.derived());
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
  return for_each_element_with_short_circuit(
      order_flag<unordered>(), [](auto &&e) { return !!e; }, t.derived());
}

// bool any(s)
template <class ShapeT, class ET, class T>
constexpr bool any_of(const tensor_base<ShapeT, ET, T> &t) {
  return !for_each_element_with_short_circuit(
      order_flag<unordered>(), [](auto &&e) { return !e; }, t.derived());
}
}
