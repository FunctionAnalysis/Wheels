#pragma once

#include <cassert>

#include "tensor_base_fwd.hpp"

#include "aligned_fwd.hpp"
#include "block_fwd.hpp"
#include "cartesian_fwd.hpp"
#include "cat_fwd.hpp"
#include "constants_fwd.hpp"
#include "diagonal_fwd.hpp"
#include "downgrade_fwd.hpp"
#include "ewise_fwd.hpp"
#include "index_fwd.hpp"
#include "iota_fwd.hpp"
#include "map_fwd.hpp"
#include "permute_fwd.hpp"
#include "remap_fwd.hpp"
#include "reshape_fwd.hpp"
#include "tensor_fwd.hpp"
#include "upgrade_fwd.hpp"

#include "const_expr.hpp"
#include "const_ints.hpp"
#include "iterators.hpp"
#include "object.hpp"
#include "parallel.hpp"
#include "types.hpp"

#include "shape.hpp"

namespace wheels {

// tags
namespace tags {
static const auto first = const_int<0>();
static const auto length = const_arg<0>();
static const auto last = length - const_int<1>();
}

namespace details {

template <class T, class E, class SizeT>
constexpr decltype(auto) _eval_index_expr_impl(T &&e, const const_expr_base<E> &,
                                          const SizeT &sz) {
  return invoke_const_expr(std::forward<T>(e), sz);
}
template <class T, class E, class SizeT>
constexpr T &&_eval_index_expr_impl(T &&t, const tensor_core<E> &,
                               const SizeT &) {
  return static_cast<T &&>(t);
}
template <class T, class E, class SizeT>
constexpr T &&_eval_index_expr_impl(T &&t, const category::other<E> &,
                               const SizeT &) {
  return static_cast<T &&>(t);
}

// _eval_index_expr
// const_expr -> invoke with sz
// other -> pass
template <class T, class SizeT>
constexpr decltype(auto) _eval_index_expr(T &&e, const SizeT &sz) {
  return _eval_index_expr_impl(std::forward<T>(e), category::identify(e), sz);
}

// _brackets
template <class T, class E, class EE>
constexpr decltype(auto) _brackets_impl(T &&t, const category::other<E> &,
                                        EE &&ind) {
  return element_at_index(std::forward<T>(t), std::forward<EE>(ind));
}

template <class T, class TensorT, class TensorTT>
constexpr auto _brackets_impl(T &&t, const tensor_core<TensorT> &id,
                              TensorTT &&inds) {
  return ::wheels::at_indices(std::forward<T>(t), std::forward<TensorTT>(inds));
}

template <class T, class TensorTT>
constexpr decltype(auto) _brackets(T &&t, TensorTT &&inds) {
  return _brackets_impl(std::forward<T>(t), category::identify(inds),
                        std::forward<TensorTT>(inds));
}

// _all_as_tensor
template <class E, class EE>
constexpr auto _all_as_tensor_impl(const category::other<E> &id, EE &&s) {
  return ::wheels::constants(make_shape(), std::forward<EE>(s));
}
template <class TensorT, class TensorTT>
constexpr TensorTT &&_all_as_tensor_impl(const tensor_core<TensorT> &id,
                                         TensorTT &&inds) {
  return static_cast<TensorTT &&>(inds);
}
template <class T>
constexpr auto _all_as_tensor(T &&t) {
  return _all_as_tensor_impl(category::identify(t), std::forward<T>(t));
}

// _block_seq
template <class T, size_t... Is, class... SubsTensorOrIntTs>
constexpr auto _block_seq(T &&t, const const_ints<size_t, Is...> &,
                          SubsTensorOrIntTs &&... subs) {
  return ::wheels::at_block(std::forward<T>(t),
                            _all_as_tensor(_eval_index_expr(
                                std::forward<SubsTensorOrIntTs>(subs),
                                (int64_t)size_at(t, const_index<Is>())))...);
}
}

// tensor_core
template <class T> struct tensor_core : category::object<T> {
  const tensor_core &core() const { return *this; }

  constexpr auto shape() const { return shape_of(this->derived()); }
  template <class K, K Idx>
  constexpr auto size(const const_ints<K, Idx> &i) const {
    return size_at(this->derived(), i);
  }
  constexpr auto numel() const { return numel_of(this->derived()); }

  constexpr auto norm_squared() const { return norm_squared(this->derived()); }
  constexpr auto norm() const { return norm_of(this->derived()); }

  // normalized
  constexpr decltype(auto) normalized() const & {
    return ewise(this->derived()) / this->norm();
  }
  decltype(auto) normalized() && {
    return ewise(std::move(this->derived())) / this->norm();
  }

  constexpr auto sum() const { return sum_of(this->derived()); }

  // at_or(otherwisev, subs ...)
  template <class E, class... SubTs>
  constexpr decltype(auto) at_or(E &&otherwise, const SubTs &... subs) const {
    return _at_or_seq(std::forward<E>(otherwise),
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
        this->derived(),
        details::_eval_index_expr(std::forward<E>(e), this->numel()));
  }
  template <class E> decltype(auto) operator[](E &&e) & {
    return details::_brackets(
        this->derived(),
        details::_eval_index_expr(std::forward<E>(e), this->numel()));
  }
  template <class E> decltype(auto) operator[](E &&e) && {
    return details::_brackets(
        std::move(this->derived()),
        details::_eval_index_expr(std::forward<E>(e), this->numel()));
  }

  // ewise
  constexpr decltype(auto) ewised() const & { return ewise(this->derived()); }
  decltype(auto) ewised() & { return ewise(this->derived()); }
  decltype(auto) ewised() && { return ewise(std::move(this->derived())); }


  // block
  template <class... TensorOrIndexTs>
  constexpr auto block(TensorOrIndexTs &&... tois) const & {
    return details::_block_seq(this->derived(),
                               make_const_sequence_for<TensorOrIndexTs...>(),
                               std::forward<TensorOrIndexTs>(tois)...);
  }
  template <class... TensorOrIndexTs> auto block(TensorOrIndexTs &&... tois) & {
    return details::_block_seq(this->derived(),
                               make_const_sequence_for<TensorOrIndexTs...>(),
                               std::forward<TensorOrIndexTs>(tois)...);
  }
  template <class... TensorOrIndexTs>
  auto block(TensorOrIndexTs &&... tois) && {
    return details::_block_seq(std::move(this->derived()),
                               make_const_sequence_for<TensorOrIndexTs...>(),
                               std::forward<TensorOrIndexTs>(tois)...);
  }

  // for_each
  template <class FunT> void for_each(FunT fun) const & {
    for_each_element(behavior_flag<unordered>(), fun, this->derived());
  }
  template <class FunT> void for_each(FunT fun) & {
    for_each_element(behavior_flag<unordered>(), fun, this->derived());
  }
  template <class FunT> void for_each(FunT fun) && {
    for_each_element(behavior_flag<unordered>(), fun,
                     std::move(this->derived()));
  }

  // reshape
  template <class ST, class... SizeTs>
  constexpr auto reshaped(const tensor_shape<ST, SizeTs...> &ns) const & {
    return reshape(this->derived(), ns);
  }
  template <class ST, class... SizeTs>
  auto reshaped(const tensor_shape<ST, SizeTs...> &ns) & {
    return reshape(this->derived(), ns);
  }
  template <class ST, class... SizeTs>
  auto reshaped(const tensor_shape<ST, SizeTs...> &ns) && {
    return reshape(std::move(this->derived()), ns);
  }

  // vectorized
  constexpr auto vectorized() const & {
    return this->reshaped(make_shape(this->numel()));
  }
  auto vectorized() & { return this->reshaped(make_shape(this->numel())); }
  auto vectorized() && {
    return std::move(derived()).reshaped(make_shape(this->numel()));
  }

  // downgraded
  template <class K, K FixedRank>
  constexpr decltype(auto)
  downgraded(const const_ints<K, FixedRank> &r) const & {
    return downgrade(this->derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) downgraded(const const_ints<K, FixedRank> &r) & {
    return downgrade(this->derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) downgraded(const const_ints<K, FixedRank> &r) && {
    return downgrade(std::move(this->derived()), r);
  }

  // upgraded
  template <class K, K FixedRank>
  constexpr decltype(auto) upgraded(const const_ints<K, FixedRank> &r) const & {
    return upgrade(this->derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) upgraded(const const_ints<K, FixedRank> &r) & {
    return upgrade(this->derived(), r);
  }
  template <class K, K FixedRank>
  decltype(auto) upgraded(const const_ints<K, FixedRank> &r) && {
    return upgrade(std::move(this->derived()), r);
  }

  // upgraded_all
  constexpr decltype(auto) upgraded_all() const & {
    return upgrade_all(this->derived());
  }
  decltype(auto) upgraded_all() & { return upgrade_all(this->derived()); }
  decltype(auto) upgraded_all() && {
    return upgrade_all(std::move(this->derived()));
  }

  // permute
  template <class... IndexTs>
  constexpr decltype(auto) permuted(const IndexTs &... inds) const & {
    return permute(this->derived(), inds...);
  }
  template <class... IndexTs>
  decltype(auto) permuted(const IndexTs &... inds) && {
    return permute(std::move(this->derived()), inds...);
  }

  // all
  constexpr bool all() const { return all_of(this->derived()); }
  // any
  constexpr bool any() const { return any_of(this->derived()); }
  // none
  constexpr bool none() const { return !any_of(this->derived()); }


  // begin/end
  constexpr tensor_iterator<const T> begin() const {
    return tensor_iterator<const T>(this->derived(), 0);
  }
  constexpr tensor_iterator<const T> end() const {
    return tensor_iterator<const T>(this->derived(), this->numel());
  }
  tensor_iterator<T> begin() { return tensor_iterator<T>(this->derived(), 0); }
  tensor_iterator<T> end() {
    return tensor_iterator<T>(this->derived(), this->numel());
  }

private:
  template <class... SubEs, size_t... Is>
  constexpr bool _valid_subs_seq(const_ints<size_t, Is...> seq,
                                 const SubEs &... subes) const {
    return ::wheels::all(::wheels::is_between(
        details::_eval_index_expr(subes, this->size(const_size<Is>())), 0,
        size(const_size<Is>()))...);
  }
  template <class E, class... SubEs, size_t... Is>
  constexpr decltype(auto) _at_or_seq(E &&otherwise,
                                      const_ints<size_t, Is...> seq,
                                      const SubEs &... subes) const {
    return _valid_subs_seq(seq, subes...) ? _parenthesis_seq(seq, subes...)
                                          : std::forward<E>(otherwise);
  }

  template <class... SubEs, size_t... Is>
  constexpr decltype(auto) _parenthesis_seq(const_ints<size_t, Is...> seq,
                                            const SubEs &... subes) const {
    assert(_valid_subs_seq(seq, subes...));
    return element_at(
        this->derived(), details::_eval_index_expr(subes, size(const_size<Is>()))...);
  }
  template <class... SubEs, size_t... Is>
  decltype(auto) _parenthesis_seq(const_ints<size_t, Is...> seq,
                                  const SubEs &... subes) {
    assert(_valid_subs_seq(seq, subes...));
    return element_at(
        this->derived(),
        details::_eval_index_expr(subes, this->size(const_size<Is>()))...);
  }
};

template <class T1, class T2>
constexpr bool operator==(const tensor_core<T1> &a, const tensor_core<T2> &b) {
  using check_t = decltype(same_rank(a.shape(), b.shape()));
  static_assert(check_t::value, "shape rank mismatched!");
  return a.shape() == b.shape() &&
         for_each_element(behavior_flag<break_on_false>(),
                          [](auto &&e1, auto &&e2) -> bool { return e1 == e2; },
                          a.derived(), b.derived());
}
template <class T1, class T2>
constexpr bool operator!=(const tensor_core<T1> &a, const tensor_core<T2> &b) {
  using check_t = decltype(same_rank(a.shape(), b.shape()));
  static_assert(check_t::value, "shape rank mismatched!");
  return !(a == b);
}

// tensor_iterator
template <class T> struct tensor_iterator {
  T &self;
  ptrdiff_t ind;
  constexpr tensor_iterator(T &s, ptrdiff_t i) : self(s), ind(i) {}
  constexpr decltype(auto) operator*() const {
    return element_at_index(self, ind);
  }
  constexpr decltype(auto) operator-> () const {
    return element_at_index(self, ind);
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

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
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

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
  constexpr operator tensor_type() const { return eval(); }

  constexpr operator value_type() const { return element_at(this->derived()); }
};

// -- necessary tensor functions
// Shape shape_of(ts);
template <class T>
constexpr tensor_shape<size_t> shape_of(const tensor_core<T> &) {
  static_assert(always<bool, false, T>::value,
                "shape_of(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
  return tensor_shape<size_t>();
}

// Scalar element_at(ts, subs ...);
template <class T, class... SubTs>
constexpr double element_at(const tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
  return 0.0;
}
template <class T, class... SubTs>
inline double &element_at(tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(T &) is not supported by tensor_core<T>, do you "
                "forget to call .derived()?");
  static double _dummy = 123.45;
  return _dummy;
}

// -- auxiliary tensor functions
// auto size_at(ts, const_int);
template <class T, class ST, ST Idx>
constexpr auto size_at(const tensor_core<T> &t,
                       const const_ints<ST, Idx> &idx) {
  return shape_of(t.derived()).at(idx);
}

// auto numel_of(ts)
template <class T> constexpr auto numel_of(const tensor_core<T> &t) {
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
template <class T, class IndexT>
decltype(auto) element_at_index(tensor_core<T> &t, const IndexT &ind) {
  return invoke_with_subs(shape_of(t.derived()), ind, [&t](auto &&... subs) {
    return element_at(t.derived(), subs...);
  });
}

// void reserve_shape(ts, shape);
template <class T, class ST, class... SizeTs>
void reserve_shape(tensor_core<T> &, const tensor_shape<ST, SizeTs...> &shape) {
}

// index_ascending
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for_each_subscript(t.shape(), [&](auto &&... subs) {
    fun(element_at(t.derived(), subs...), element_at(ts.derived(), subs...)...);
  });
  return true;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT fun,
                      tensor_core<T> &t, Ts &&... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for_each_subscript(t.shape(), [&](auto &&... subs) {
    fun(element_at(t.derived(), subs...), element_at(ts.derived(), subs...)...);
  });
  return true;
}

// unordered
namespace details {
template <class FunT, class T, class... Ts>
void _for_each_element_unordered_default(yes staticShape, FunT fun, T &&t,
                                         Ts &&... ts) {
  for_each_element(behavior_flag<index_ascending>(), fun, std::forward<T>(t),
                   std::forward<Ts>(ts)...);
}

static constexpr size_t _numel_parallel_thres = (size_t)4e4;
template <class FunT, class T, class... Ts>
void _for_each_element_unordered_parallel(FunT &&fun, T &&t, Ts &&... ts) {
  parallel_for_each((size_t)(numel_of(t)),
                    [&fun, &t, &ts...](size_t i) {
                      fun(element_at_index(t.derived(), i),
                          element_at_index(ts.derived(), i)...);
                    },
                    _numel_parallel_thres / 2);
}

template <class FunT, class T, class... Ts>
void _for_each_element_unordered_default(no staticShape, FunT fun, T &&t,
                                         Ts &&... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  if (t.numel() < _numel_parallel_thres) {
    for_each_element(behavior_flag<index_ascending>(), fun, std::forward<T>(t),
                     std::forward<Ts>(ts)...);
  } else {
    _for_each_element_unordered_parallel(fun, std::forward<T>(t),
                                         std::forward<Ts>(ts)...);
  }
}
}
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts) {
  details::_for_each_element_unordered_default(
      const_bool<std::decay_t<decltype(t.shape())>::is_static>(), fun,
      t.derived(), ts.derived()...);
  return true;
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT fun, tensor_core<T> &t,
                      Ts &&... ts) {
  details::_for_each_element_unordered_default(
      const_bool<std::decay_t<decltype(t.shape())>::is_static>(), fun,
      t.derived(), ts.derived()...);
  return true;
}

// break_on_false
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  return for_each_subscript_if(t.shape(), [&](auto &&... subs) {
    return fun(element_at(t.derived(), subs...),
               element_at(ts.derived(), subs...)...);
  });
}

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      tensor_core<T> &t, Ts &&... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  return for_each_subscript_if(t.shape(), [&](auto &&... subs) {
    return fun(element_at(t.derived(), subs...),
               element_at(ts.derived(), subs...)...);
  });
}

// nonzero_only
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts) {
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
bool for_each_element(behavior_flag<nonzero_only>, FunT fun, tensor_core<T> &t,
                      Ts &&... ts) {
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

// void assign_elements(to, from);
template <class ToT, class FromT>
void assign_elements(tensor_core<ToT> &to, const tensor_core<FromT> &from) {
  using _shape_to = std::decay_t<decltype(to.shape())>;
  using _shape_from = std::decay_t<decltype(from.shape())>;
  static_assert(_shape_to::rank == _shape_from::rank, "shape ranks mismatch!");

  decltype(auto) s = from.shape();
  if (to.shape() != s) {
    reserve_shape(to.derived(), s);
  }
  for_each_element(behavior_flag<unordered>(),
                   [](auto &&to_e, auto &&from_e) {
                     auto e = from_e;
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
E reduce_elements(const tensor_core<T> &t, E initial, ReduceT &&red) {
  for_each_element(behavior_flag<unordered>(),
                   [&initial, &red](auto &&e) { initial = red(initial, e); },
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

// Scalar norm_of(ts)
template <class ET, class ShapeT, class T>
constexpr auto norm_of(const tensor_base<ET, ShapeT, T> &t) {
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
