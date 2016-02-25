#pragma once

#include <cassert>

#include "../core/const_expr.hpp"
#include "../core/const_ints.hpp"
#include "../core/iterators.hpp"
#include "../core/overloads.hpp"
#include "../core/parallel.hpp"
#include "../core/types.hpp"

#include "shape.hpp"

namespace wheels {

template <class EleT, class ShapeT, class T> struct category_tensor {};

template <class T> struct tensor_core;
template <class T> struct tensor_iterator;

// index_tags
namespace index_tags {
constexpr auto first = const_index<0>();
constexpr auto length = const_symbol<0>();
constexpr auto last = length - const_index<1>();
constexpr auto everything = make_interval(first, length);
}

namespace details {
template <class E, class SizeT,
          class = std::enable_if_t<is_const_expr<std::decay_t<E>>::value>>
constexpr auto _eval_index_expr(E &&e, const SizeT &sz) {
  return forward<E>(e)(sz);
}
template <class T, class SizeT,
          class = std::enable_if_t<!is_const_expr<std::decay_t<T>>::value>,
          class = void>
constexpr T &&_eval_index_expr(T &&t, const SizeT &) {
  return static_cast<T &&>(t);
}

// _brackets
template <class T, class E, class EE,
          class = std::enable_if_t<is_int<E>::value>>
constexpr decltype(auto) _brackets_impl(T &&t, const E &, EE &&ind) {
  return ::wheels::element_at_index(forward<T>(t), forward<EE>(ind));
}

template <class T, class TensorT, class TensorTT>
constexpr auto _brackets_impl(T &&t, const tensor_core<TensorT> &,
                              TensorTT &&inds) {
  return ::wheels::at_indices(forward<T>(t), forward<TensorTT>(inds));
}

template <class T, class TensorTT>
constexpr decltype(auto) _brackets(T &&t, TensorTT &&inds) {
  return _brackets_impl(forward<T>(t), inds, forward<TensorTT>(inds));
}

//
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

  // operator[](index/{index tensor})
  template <class E> constexpr decltype(auto) operator[](E &&e) const & {
    return details::_brackets(
        derived(), details::_eval_index_expr(forward<E>(e), numel()));
  }
  template <class E> decltype(auto) operator[](E &&e) & {
    return details::_brackets(
        derived(), details::_eval_index_expr(forward<E>(e), numel()));
  }
  template <class E> decltype(auto) operator[](E &&e) && {
    return details::_brackets(
        move(derived()), details::_eval_index_expr(forward<E>(e), numel()));
  }

  // for_each
  template <class FunT> void for_each(FunT &fun) const {
    ::wheels::for_each_element(behavior_flag<unordered>(), fun, derived());
  }
  template <class FunT> void for_each(FunT &fun) {
    ::wheels::for_each_element(behavior_flag<unordered>(), fun, derived());
  }

  // transform
  template <class FunT> constexpr auto transform(FunT &&fun) const & {
    return ::wheels::transform(derived(), forward<FunT>(fun));
  }
  template <class FunT> auto transform(FunT &&fun) && {
    return ::wheels::transform(std::move(derived()), forward<FunT>(fun));
  }

  // reshape
  template <class ST, class... SizeTs>
  constexpr auto reshape(const tensor_shape<ST, SizeTs...> &ns) const & {
    return ::wheels::reshape(derived(), ns);
  }
  template <class ST, class... SizeTs>
  auto reshape(const tensor_shape<ST, SizeTs...> &ns) & {
    return ::wheels::reshape(derived(), ns);
  }
  template <class ST, class... SizeTs>
  auto reshape(const tensor_shape<ST, SizeTs...> &ns) && {
    return ::wheels::reshape(std::move(derived()), ns);
  }

  // subwise
  template <class K, K FixedRank>
  constexpr decltype(auto) subwise(const const_ints<K, FixedRank> &r) const & {
    return ::wheels::subwise(derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) subwise(const const_ints<K, FixedRank> &r) & {
    return ::wheels::subwise(derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) subwise(const const_ints<K, FixedRank> &r) && {
    return ::wheels::subwise(std::move(derived()), r);
  }

  // permute
  template <class... IndexTs>
  constexpr decltype(auto) permute(const IndexTs &... inds) const & {
    return ::wheels::permute(derived(), inds...);
  }
  template <class... IndexTs>
  decltype(auto) permute(const IndexTs &... inds) && {
    return ::wheels::permute(std::move(derived()), inds...);
  }

  // block
  template <class... RangeOrIndexTs>
  constexpr auto block(const RangeOrIndexTs &... rois) const & {
    return ::wheels::block_at(derived(), rois...);
  }
  template <class... RangeOrIndexTs>
  auto block(const RangeOrIndexTs &... rois) & {
    return ::wheels::block_at(derived(), rois...);
  }
  template <class... RangeOrIndexTs>
  auto block(const RangeOrIndexTs &... rois) && {
    return ::wheels::block_at(std::move(derived()), rois...);
  }

  // begin/end
  constexpr tensor_iterator<const T> begin() const {
    return tensor_iterator<const T>(derived(), 0);
  }
  constexpr tensor_iterator<const T> end() const {
    return tensor_iterator<const T>(derived(), numel());
  }
  tensor_iterator<T> begin() { return tensor_iterator<T>(derived(), 0); }
  tensor_iterator<T> end() { return tensor_iterator<T>(derived(), numel()); }

private:
  template <class... SubEs, size_t... Is>
  constexpr bool _valid_subs_seq(const_ints<size_t, Is...> seq,
                                 const SubEs &... subes) const {
    return ::wheels::all(::wheels::is_between(
        details::_eval_index_expr(subes, size(const_size<Is>())), 0,
        size(const_size<Is>()))...);
  }
  template <class E, class... SubEs, size_t... Is>
  constexpr decltype(auto) _at_or_seq(E &&otherwise,
                                      const_ints<size_t, Is...> seq,
                                      const SubEs &... subes) const {
    return _valid_subs_seq(seq, subes...) ? _parenthesis_seq(seq, subes...)
                                          : forward<E>(otherwise);
  }

  template <class... SubEs, size_t... Is>
  constexpr decltype(auto) _parenthesis_seq(const_ints<size_t, Is...> seq,
                                            const SubEs &... subes) const {
    assert(_valid_subs_seq(seq, subes...));
    return ::wheels::element_at(
        derived(), details::_eval_index_expr(subes, size(const_size<Is>()))...);
  }
  template <class... SubEs, size_t... Is>
  decltype(auto) _parenthesis_seq(const_ints<size_t, Is...> seq,
                                  const SubEs &... subes) {
    assert(_valid_subs_seq(seq, subes...));
    return ::wheels::element_at(
        derived(), details::_eval_index_expr(subes, size(const_size<Is>()))...);
  }
};

// tensor_iterator
template <class T> struct tensor_iterator {
  T &self;
  ptrdiff_t ind;
  constexpr tensor_iterator(T &s, ptrdiff_t i) : self(s), ind(i) {}
  constexpr decltype(auto) operator*() const {
    return ::wheels::element_at_index(self, ind);
  }
  constexpr decltype(auto) operator-> () const {
    return ::wheels::element_at_index(self, ind);
  }
  tensor_iterator &operator++() {
    ++ind;
    return *this;
  }
  tensor_iterator operator++(int) {
    auto i = *this;
    ++ind;
    return i;
  }
  constexpr bool operator==(const tensor_iterator &i) const {
    assert(&self == &(i.self));
    return ind == i.ind;
  }
  constexpr bool operator!=(const tensor_iterator &i) const {
    assert(&self == &(i.self));
    return ind != i.ind;
  }
};

template <class ET, class ShapeT> class tensor;

// tensor_base<ET, ShapeT, T>
template <class ET, class ShapeT, class T> struct tensor_base : tensor_core<T> {
  using value_type = ET;
  using shape_type = ShapeT;
  static constexpr size_t rank = ShapeT::rank;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(derived()); }
  tensor_type eval() && { return tensor_type(move(derived())); }
  constexpr operator tensor_type() const { return eval(); }
};

// 0 dimensional tensor (scalar)
template <class ET, class ST, class T>
struct tensor_base<ET, tensor_shape<ST>, T> : tensor_core<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST>;
  static constexpr size_t rank = 0;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(derived()); }
  tensor_type eval() && { return tensor_type(move(derived())); }
  constexpr operator tensor_type() const { return eval(); }

  constexpr operator value_type() const {
    return ::wheels::element_at(derived());
  }
};

// 1 dimensional tensor (vector)
template <class ET, class ST, class NT, class T>
struct tensor_base<ET, tensor_shape<ST, NT>, T> : tensor_core<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST, NT>;
  static constexpr size_t rank = 1;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(derived()); }
  tensor_type eval() && { return tensor_type(move(derived())); }
  constexpr operator tensor_type() const { return eval(); }

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
  dot(const tensor_base<ET2, tensor_shape<ST2, NT2>, T2> &t) const {
    return ::wheels::dot(*this, t);
  }
  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  cross(const tensor_base<ET2, tensor_shape<ST2, NT2>, T2> &t) const {
    return ::wheels::cross(*this, t);
  }
};

// 2 dimensional tensor (matrix)
template <class ET, class ST, class MT, class NT, class T>
struct tensor_base<ET, tensor_shape<ST, MT, NT>, T> : tensor_core<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST, MT, NT>;
  static constexpr size_t rank = 2;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(derived()); }
  tensor_type eval() && { return tensor_type(move(derived())); }
  constexpr operator tensor_type() const { return eval(); }

  constexpr auto rows() const { return size(const_index<0>()); }
  constexpr auto cols() const { return size(const_index<1>()); }

  constexpr decltype(auto) t() const & {
    return ::wheels::transpose(derived());
  }
  decltype(auto) t() & { return ::wheels::transpose(derived()); }
  decltype(auto) t() && { return ::wheels::transpose(std::move(derived())); }
};

// category_for_overloading
// common_func
template <class ET, class ShapeT, class T, class OpT>
constexpr auto category_for_overloading(const tensor_base<ET, ShapeT, T> &,
                                        const common_func<OpT> &) {
  return category_tensor<ET, ShapeT, T>();
}

// tensor_op_result_base
template <class EleT, class ShapeT, class OpT, class T>
struct tensor_op_result_base : tensor_base<EleT, ShapeT, T> {};

// t1 == t2
template <class ShapeT, class T>
struct tensor_op_result_base<bool, ShapeT, binary_op_eq, T>
    : tensor_base<bool, ShapeT, T> {
  constexpr operator bool() const {
    return ::wheels::equals_result_of(derived());
  }
};

// t1 != t2
template <class ShapeT, class T>
struct tensor_op_result_base<bool, ShapeT, binary_op_neq, T>
    : tensor_base<bool, ShapeT, T> {
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
  for_each_subscript(t.shape(), [&](auto &&... subs) {
    decltype(auto) e = element_at(t.derived(), subs...);
    if (!is_zero(e)) {
      fun(e, element_at(ts.derived(), subs...)...);
    } else {
      visited_all = false;
    }
  });
  return visited_all;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun, tensor_core<T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  bool visited_all = true;
  for_each_subscript(t.shape(), [&](auto &... subs) {
    decltype(auto) e = element_at(t.derived(), subs...);
    if (!is_zero(e)) {
      fun(e, element_at(ts.derived(), subs...)...);
    } else {
      visited_all = false;
    }
  });
  return visited_all;
}

// void assign_elements(to, from);
template <class ToET, class ToShapeT, class ToT, class FromET, class FromShapeT,
          class FromT>
void assign_elements(tensor_base<ToET, ToShapeT, ToT> &to,
                     const tensor_base<FromET, FromShapeT, FromT> &from) {
  decltype(auto) s = from.shape();
  if (to.shape() != s) {
    reserve_shape(to.derived(), s);
  }
  for_each_element(behavior_flag<unordered>(),
                   [](auto &&to_e, auto &&from_e) {
                     FromET e = from_e;
                     to_e = e;
                   },
                   to.derived(), from.derived());
}

// void fill_elements_with(to, scalar)
template <class T, class E>
void fill_elements_with(tensor_core<T> &t, const E &e) {
  for_each_element(behavior_flag<unordered>(), [&e](auto &&te) { te = e; },
                   t.derived());
}

// size_t nonzero_elements_count(t)
template <class T> size_t nonzero_elements_count(const tensor_core<T> &t) {
  size_t nzc = 0;
  for_each_element(behavior_flag<nonzero_only>(), [&nzc](auto &&e) { nzc++; },
                   t.derived());
  return nzc;
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
template <class ET, class ShapeT, class T>
ET norm_squared(const tensor_base<ET, ShapeT, T> &t) {
  auto result = types<ET>::zero();
  for_each_element(behavior_flag<nonzero_only>(),
                   [&result](auto &&e) { result += e * e; }, t.derived());
  return result;
}

// Scalar norm(ts)
template <class ET, class ShapeT, class T>
constexpr auto norm(const tensor_base<ET, ShapeT, T> &t) {
  return sqrt(norm_squared(t.derived()));
}

// bool all(s)
template <class ET, class ShapeT, class T>
constexpr bool all_of(const tensor_base<ET, ShapeT, T> &t) {
  return for_each_element(behavior_flag<break_on_false>(),
                          [](auto &&e) { return !!e; }, t.derived());
}

// bool any(s)
template <class ET, class ShapeT, class T>
constexpr bool any_of(const tensor_base<ET, ShapeT, T> &t) {
  return !for_each_element(behavior_flag<break_on_false>(),
                           [](auto &&e) { return !e; }, t.derived());
}

// equals_result_of
template <class ET, class ShapeT, class T>
constexpr bool equals_result_of(const tensor_base<ET, ShapeT, T> &t) {
  return all_of(t);
}

// not_equals_result_of
template <class ET, class ShapeT, class T>
constexpr bool not_equals_result_of(const tensor_base<ET, ShapeT, T> &t) {
  return any_of(t);
}

// Scalar sum(s)
template <class ET, class ShapeT, class T>
ET sum_of(const tensor_base<ET, ShapeT, T> &t) {
  auto s = types<ET>::zero();
  for_each_element(behavior_flag<nonzero_only>(), [&s](auto &&e) { s += e; },
                   t.derived());
  return s;
}

// ostream
namespace details {
template <class ET, class ShapeT, class T>
inline std::ostream &_stream_impl(std::ostream &os,
                                  const tensor_base<ET, ShapeT, T> &t,
                                  const_size<0>) {
  return os << t();
}
template <class ET, class ShapeT, class T>
inline std::ostream &_stream_impl(std::ostream &os,
                                  const tensor_base<ET, ShapeT, T> &t,
                                  const_size<1>) {
  static const char _bracket[2] = {conditional(is_character<ET>(), '\"', '['),
                                   conditional(is_character<ET>(), '\"', ']')};
  if (t.numel() == 0) {
    return os << _bracket[0] << _bracket[1];
  }
  os << _bracket[0] << t(0);
  for (size_t i = 1; i < t.numel(); i++) {
    auto e = t(i);
    if (!is_character<ET>()) {
      os << ", ";
    }
    os << e;
  }
  return os << _bracket[1];
}

template <class ET, class ShapeT, class T>
inline std::ostream &_stream_impl(std::ostream &os,
                                  const tensor_base<ET, ShapeT, T> &t,
                                  const_size<2>) {
  static const char _bracket[2] = {conditional(is_character<ET>(), '\"', '['),
                                   conditional(is_character<ET>(), '\"', ']')};
  for (size_t j = 0; j < t.size(const_index<0>()); j++) {
    if (t.size(const_index<1>()) == 0) {
      os << _bracket[0] << _bracket[1] << '\n';
    } else {
      os << _bracket[0] << t(j, 0);
      for (size_t i = 1; i < t.size(const_index<1>()); i++) {
        if (!is_character<ET>()) {
          os << ", ";
        }
        os << t(j, i);
      }
      os << _bracket[1] << '\n';
    }
  }
  return os;
}
template <class ET, class ShapeT, class T, size_t I>
inline std::ostream &_stream_impl(std::ostream &os,
                                  const tensor_base<ET, ShapeT, T> &t,
                                  const_size<I>) {
  static_assert(always<bool, false, ShapeT>::value, "not implemented yet");
  return os;
}
}
template <class ET, class ShapeT, class T>
inline std::ostream &operator<<(std::ostream &os,
                                const tensor_base<ET, ShapeT, T> &t) {
  return details::_stream_impl(os, t.derived(), const_size<ShapeT::rank>());
}

// is_zero
template <class ET, class ShapeT, class T>
bool is_zero(const tensor_base<ET, ShapeT, T> &t) {
  return for_each_element(behavior_flag<break_on_false>(),
                          [](auto &&e) { return is_zero(e); }, t.derived());
}
}
