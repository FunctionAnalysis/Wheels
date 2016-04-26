#pragma once

#include "../core/const_ints.hpp"
#include "../core/utility.hpp"

#include "shape_fwd.hpp"

namespace wheels {

// tensor_shape
template <class T> class tensor_shape<T> {
  static_assert(std::is_integral<T>::value, "T should be an integral type");

public:
  using value_type = T;
  static constexpr size_t rank = 0;
  static constexpr bool is_static = true;
  static constexpr T static_magnitude = 1;
  static constexpr size_t static_size_num = 0;
  static constexpr size_t dynamic_size_num = 0;
  static constexpr ptrdiff_t last_dynamic_dim = -1;

  constexpr const_ints<T, 1> magnitude() const { return const_ints<T, 1>(); }

  constexpr tensor_shape() {}
  template <class K> constexpr tensor_shape(const tensor_shape<K> &) {}

  template <class K> tensor_shape part(const const_ints<K> &) const {
    return tensor_shape();
  }
};

template <class T, T S, class... SizeTs>
class tensor_shape<T, const_ints<T, S>, SizeTs...>
    : public tensor_shape<T, SizeTs...> {
  static_assert(std::is_integral<T>::value, "T should be an integral type");
  using this_t = tensor_shape<T, const_ints<T, S>, SizeTs...>;
  using rest_tensor_shape_t = tensor_shape<T, SizeTs...>;

public:
  using value_type = T;
  static constexpr size_t rank = sizeof...(SizeTs) + 1;
  static constexpr bool is_static = rest_tensor_shape_t::is_static;
  static constexpr T static_value = S;
  static constexpr T static_magnitude =
      S * rest_tensor_shape_t::static_magnitude;
  static constexpr size_t static_size_num =
      rest_tensor_shape_t::static_size_num + 1;
  static constexpr size_t dynamic_size_num =
      rest_tensor_shape_t::dynamic_size_num;
  static constexpr ptrdiff_t last_dynamic_dim =
      rest_tensor_shape_t::last_dynamic_dim == -1
          ? -1
          : (rest_tensor_shape_t::last_dynamic_dim + 1);

  const rest_tensor_shape_t &rest() const {
    return (const rest_tensor_shape_t &)(*this);
  }
  rest_tensor_shape_t &rest() { return (rest_tensor_shape_t &)(*this); }
  rest_tensor_shape_t &&rest_rref() { return (rest_tensor_shape_t &&)(*this); }

  constexpr const_ints<T, S> value() const { return const_ints<T, S>(); }
  constexpr auto magnitude() const { return value() * rest().magnitude(); }

  // ctor
  constexpr tensor_shape() : rest_tensor_shape_t() {}

  // ctor from vals
  template <class... Ks>
  constexpr explicit tensor_shape(const const_ints<T, S> &, const Ks &... vals)
      : rest_tensor_shape_t(vals...) {}
  template <class K, class... Ks>
  constexpr explicit tensor_shape(const K &v, const Ks &... vals)
      : rest_tensor_shape_t(vals...) {
    static_assert(is_int<K>::value, "T must be an integer type");
  }
  template <class... Ks>
  constexpr explicit tensor_shape(ignore_t, const Ks &... vals)
      : rest_tensor_shape_t(vals...) {}

  // ctor from tensor_shape
  template <class K, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  constexpr tensor_shape(const tensor_shape<K, const_ints<K, S>, SizeT2s...> &t)
      : rest_tensor_shape_t(t.rest()) {}
  template <class K, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  constexpr tensor_shape(const tensor_shape<K, K, SizeT2s...> &t)
      : rest_tensor_shape_t(t.rest()) {}

  // =
  template <class K, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  tensor_shape &
  operator=(const tensor_shape<K, const_ints<K, S>, SizeT2s...> &t) {
    rest() = t.rest();
    return *this;
  }
  template <class K, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  tensor_shape &operator=(const tensor_shape<K, K, SizeT2s...> &t) {
    assert(t.value() == S);
    rest() = t.rest();
    return *this;
  }

  // copy ctor
  constexpr tensor_shape(const tensor_shape &) = default;
  tensor_shape(tensor_shape &&) = default;
  tensor_shape &operator=(const tensor_shape &) = default;
  tensor_shape &operator=(tensor_shape &&) = default;

  // at
  template <size_t Idx> constexpr auto at(const const_index<Idx> &) const {
    static_assert(Idx < rank, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }
  constexpr auto at(const const_index<0> &) const { return value(); }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  constexpr auto at(const const_ints<K, Idx> &) const {
    return at(const_index<Idx>());
  }
  template <class K, K Idx>
  constexpr auto operator[](const const_ints<K, Idx> &i) const {
    return at(i);
  }

  // part
  template <class K, K I, K... Is>
  constexpr auto part(const const_ints<K, I, Is...> &is) const {
    return make_shape(at(const_index<I>()), at(const_index<Is>())...);
  }

  // resize
  template <size_t Idx> void resize(const const_index<Idx> &, T ns) {
    rest().resize(const_index<Idx - 1>(), ns);
  }
  void resize(const const_index<0u> &, T ns) { assert(ns == S); }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  void resize(const const_ints<K, Idx> &, T ns) {
    resize(const_index<Idx>(), ns);
  }

  // mag_at
  template <size_t Idx> constexpr auto mag_at(const const_index<Idx> &) const {
    static_assert(Idx < rank, "Idx too large");
    return rest().mag_at(const_index<Idx - 1>());
  }
  constexpr auto mag_at(const const_index<0> &) const { return magnitude(); }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  constexpr auto mag_at(const const_ints<K, Idx> &) const {
    return mag_at(const_index<Idx>());
  }

  template <class Archive> void serialize(Archive &ar) {
    T val = value(), mag = magnitude();
    ar(val, mag);
    ar(rest());
  }
  template <class V> decltype(auto) fields(V &&v) { return v(rest()); }
  template <class V> constexpr decltype(auto) fields(V &&v) const {
    return v(rest());
  }
};

template <class T, class... SizeTs>
class tensor_shape<T, T, SizeTs...> : public tensor_shape<T, SizeTs...> {
  static_assert(std::is_integral<T>::value, "T should be an integral type");
  using this_t = tensor_shape<T, T, SizeTs...>;
  using rest_tensor_shape_t = tensor_shape<T, SizeTs...>;

public:
  using value_type = T;
  static constexpr size_t rank = sizeof...(SizeTs) + 1;
  static constexpr bool is_static = false;
  static constexpr T static_value = 1;
  static constexpr T static_magnitude = rest_tensor_shape_t::static_magnitude;
  static constexpr size_t static_size_num =
      rest_tensor_shape_t::static_size_num;
  static constexpr size_t dynamic_size_num =
      rest_tensor_shape_t::dynamic_size_num + 1;
  static constexpr ptrdiff_t last_dynamic_dim = 0;

  constexpr const rest_tensor_shape_t &rest() const {
    return (const rest_tensor_shape_t &)(*this);
  }
  rest_tensor_shape_t &rest() { return (rest_tensor_shape_t &)(*this); }
  rest_tensor_shape_t &&rest_rref() { return (rest_tensor_shape_t &&)(*this); }

  constexpr T value() const { return _val; }
  constexpr T magnitude() const { return _mag; }

  // ctor
  constexpr tensor_shape() : rest_tensor_shape_t(), _val(0), _mag(0) {}

  // ctor from vals
  template <class K, class... Ks>
  constexpr explicit tensor_shape(const K &v, const Ks &... vals)
      : rest_tensor_shape_t(vals...), _val(v), _mag(v * rest().magnitude()) {
    static_assert(is_int<K>::value, "T must be an integer type");
  }

  // ctor from tensor_shape
  template <class K, class ST, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  constexpr tensor_shape(const tensor_shape<K, ST, SizeT2s...> &t)
      : rest_tensor_shape_t(t.rest()), _val(t.value()), _mag(t.magnitude()) {}

  // =
  template <class K, class ST, class... SizeT2s,
            class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
  tensor_shape &operator=(const tensor_shape<K, ST, SizeT2s...> &t) {
    _val = t.value();
    _mag = t.magnitude();
    rest() = t.rest();
    return *this;
  }

  // copy ctor
  constexpr tensor_shape(const tensor_shape &) = default;
  tensor_shape(tensor_shape &&) = default;
  tensor_shape &operator=(const tensor_shape &) = default;
  tensor_shape &operator=(tensor_shape &&) = default;

  // at
  template <size_t Idx> constexpr auto at(const const_index<Idx> &) const {
    static_assert(Idx < rank, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }
  constexpr T at(const const_index<0u>) const { return _val; }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  constexpr auto at(const const_ints<K, Idx> &) const {
    return at(const_index<Idx>());
  }
  template <class K, K Idx>
  constexpr auto operator[](const const_ints<K, Idx> &i) const {
    return at(i);
  }

  // part
  template <class K, K I, K... Is>
  constexpr auto part(const const_ints<K, I, Is...> &is) const {
    return make_shape(at(const_index<I>()), at(const_index<Is>())...);
  }

  // resize
  template <size_t Idx> void resize(const const_index<Idx> &, T ns) {
    rest().resize(const_index<Idx - 1>(), ns);
    _mag = _val * rest().magnitude();
  }
  void resize(const const_index<0u> &, T ns) {
    _val = ns;
    _mag = _val * rest().magnitude();
  }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  void resize(const const_ints<K, Idx> &, size_t ns) {
    resize(const_index<Idx>(), ns);
  }

  // mag_at
  template <size_t Idx> constexpr auto mag_at(const const_index<Idx> &) const {
    static_assert(Idx < rank, "Idx too large");
    return rest().mag_at(const_index<Idx - 1>());
  }
  constexpr auto mag_at(const const_index<0> &) const { return magnitude(); }
  template <class K, K Idx, wheels_enable_if(!(std::is_same<K, size_t>::value))>
  constexpr auto mag_at(const const_ints<K, Idx> &) const {
    return mag_at(const_index<Idx>());
  }

  template <class Archive> void serialize(Archive &ar) {
    ar(_val, _mag);
    ar(rest());
  }
  template <class V> decltype(auto) fields(V &&v) {
    return v(_val, _mag, rest());
  }
  template <class V> constexpr decltype(auto) fields(V &&v) const {
    return v(_val, _mag, rest());
  }

private:
  T _val;
  T _mag;
};

// is_tensor_shape
template <class T> struct is_tensor_shape : no {};
template <class T, class... SizeTs>
struct is_tensor_shape<tensor_shape<T, SizeTs...>> : yes {};

// shape_of_rank
namespace details {
template <class T, class SeqT> struct _make_shape_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_shape_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor_shape<T, always_t<T, Is>...>;
};
}
template <class T, size_t Rank>
using shape_of_rank = typename details::_make_shape_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;

// sub2ind
template <class T> constexpr T sub2ind(const tensor_shape<T> &) { return (T)0; }
template <class T, class SizeT, class... SizeTs>
constexpr T sub2ind(const tensor_shape<T, SizeT, SizeTs...> &) {
  return 0;
}
template <class T, class SizeT, class... SizeTs, class K, class... Ks>
constexpr T sub2ind(const tensor_shape<T, SizeT, SizeTs...> &shape, K sub,
                    Ks... subs) {
  return (T)(sub * shape.rest().magnitude() + sub2ind(shape.rest(), subs...));
}

// ind2sub
template <class T, class IndexT>
void ind2sub(const tensor_shape<T> &, const IndexT &ind) {}
template <class T, class IndexT, class SizeT, class... SizeTs>
void ind2sub(const tensor_shape<T, SizeT, SizeTs...> &, const IndexT &ind) {}
template <class T, class IndexT, class SizeT, class... SizeTs, class K,
          class... Ks>
void ind2sub(const tensor_shape<T, SizeT, SizeTs...> &shape, const IndexT &ind,
             K &sub, Ks &... subs) {
  const auto lm = shape.rest().magnitude();
  sub = ind / lm;
  ind2sub(shape.rest(), ind % lm, subs...);
}

// sub2ind_by_iter
template <class T, class SubsIterT>
constexpr T sub2ind_by_iter(const tensor_shape<T> &, SubsIterT subs_iter) {
  return 0;
}
template <class SubsIterT, class T, class SizeT, class... SizeTs>
T sub2ind_by_iter(const tensor_shape<T, SizeT, SizeTs...> &shape,
                  SubsIterT subs_iter) {
  const auto cur_sub = *subs_iter;
  ++subs_iter;
  return cur_sub * shape.rest().magnitude() +
         sub2ind_by_iter(shape.rest(), subs_iter);
}

// ind2sub_by_iter
template <class T, class IndexT, class SubsIterT>
void ind2sub_by_iter(const tensor_shape<T> &, const IndexT &ind,
                     SubsIterT subs_iter) {}
template <class SubsIterT, class IndexT, class T, class SizeT, class... SizeTs>
void ind2sub_by_iter(const tensor_shape<T, SizeT, SizeTs...> &shape,
                     const IndexT &ind, SubsIterT subs_iter) {
  const auto lm = shape.rest().magnitude();
  *subs_iter = ind / lm;
  ind2sub_by_iter(shape.rest(), ind % lm, ++subs_iter);
}

// invoke_with_subs
template <class T, class IndexT, class FunT, class... SubTs>
constexpr decltype(auto) invoke_with_subs(const tensor_shape<T> &shape,
                                          const IndexT &ind, FunT &&fun,
                                          const SubTs &... subs) {
  return std::forward<FunT>(fun)(subs...);
}
template <class T, class IndexT, class FunT, class SizeT, class... SizeTs,
          class... SubTs>
constexpr decltype(auto)
invoke_with_subs(const tensor_shape<T, SizeT, SizeTs...> &shape,
                 const IndexT &ind, FunT &&fun, const SubTs &... subs) {
  return invoke_with_subs(shape.rest(), ind % shape.rest().magnitude(),
                          std::forward<FunT>(fun), subs...,
                          ind / shape.rest().magnitude());
}

// for_each_subscript
template <class T, class FunT, class... Ts>
void for_each_subscript(const tensor_shape<T> &, FunT &&fun, Ts &&... args) {
  fun(args...);
}
template <class T, class SizeT, class... SizeTs, class FunT, class... Ts>
void for_each_subscript(const tensor_shape<T, SizeT, SizeTs...> &shape,
                        FunT &&fun, Ts &&... args) {
  const auto n = shape.value();
  for (T i = 0; i < n; i++) {
    for_each_subscript(shape.rest(), fun, args..., i);
  }
}

// for_each_subscript_if
template <class T, class FunT, class... Ts>
constexpr bool for_each_subscript_if(const tensor_shape<T> &, FunT &&fun,
                                     Ts &&... args) {
  return fun(args...);
}
template <class T, class SizeT, class... SizeTs, class FunT, class... Ts>
bool for_each_subscript_if(const tensor_shape<T, SizeT, SizeTs...> &shape,
                           FunT &&fun, Ts &&... args) {
  const auto n = shape.value();
  T i = 0;
  for (; i < n; i++) {
    if (!for_each_subscript_if(shape.rest(), fun, args..., i))
      break;
  }
  return i == n;
}

// for_each_subscript_until
template <class T, T Idx, class FunT, class... Ts>
std::enable_if_t<Idx == 0> for_each_subscript_until(const tensor_shape<T> &,
                                                    const const_ints<T, Idx> &,
                                                    FunT &&fun, Ts &&... args) {
  fun(args...);
}
template <class T, class SizeT, class... SizeTs, size_t Idx, class FunT,
          class... Ts>
std::enable_if_t<Idx == sizeof...(SizeTs) + 1>
for_each_subscript_until(const tensor_shape<T, SizeT, SizeTs...> &shape,
                         const const_index<Idx> &, FunT &&fun, Ts &&... args) {
  fun(args...);
}
template <class T, class SizeT, class... SizeTs, size_t Idx, class FunT,
          class... Ts>
std::enable_if_t<(Idx < sizeof...(SizeTs) + 1)>
for_each_subscript_until(const tensor_shape<T, SizeT, SizeTs...> &shape,
                         const const_index<Idx> &idx, FunT &&fun,
                         Ts &&... args) {
  const auto n = value();
  for (T i = 0; i < n; i++) {
    for_each_subscript_until(shape.rest(), idx, fun, args..., i);
  }
}

// max_shape_size
namespace details {
template <class ShapeT, size_t... Is>
constexpr auto _max_shape_size_seq(const ShapeT &shape,
                                   const_ints<size_t, Is...>) {
  return max(shape.at(const_index<Is>())...);
}
}
template <class T, class SizeT, class... SizeTs>
constexpr auto max_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape) {
  return details::_max_shape_size_seq(
      shape, make_const_sequence(const_size<1 + sizeof...(SizeTs)>()));
}

// min_shape_size
namespace details {
template <class ShapeT, size_t... Is>
constexpr auto _min_shape_size_seq(const ShapeT &shape,
                                   const_ints<size_t, Is...>) {
  return min(shape.at(const_index<Is>())...);
}
}
template <class T, class SizeT, class... SizeTs>
constexpr auto min_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape) {
  return details::_min_shape_size_seq(
      shape, make_const_sequence(const_size<1 + sizeof...(SizeTs)>()));
}

// make_rank_sequence
template <class T, class... SizeTs>
constexpr auto make_rank_sequence(const tensor_shape<T, SizeTs...> &shape) {
  return make_const_sequence(const_size<sizeof...(SizeTs)>());
}

// ==
template <class T, class K>
constexpr bool operator==(const tensor_shape<T> &, const tensor_shape<K> &) {
  return true;
}
template <class T, class SizeT, class... SizeTs, class K, class... SizeT2s>
constexpr std::enable_if_t<sizeof...(SizeTs) + 1 == sizeof...(SizeT2s), bool>
operator==(const tensor_shape<T, SizeT, SizeTs...> &shape,
           const tensor_shape<K, SizeT2s...> &b) {
  return shape.value() == b.value() && shape.rest() == b.rest();
}

// !=
template <class T1, class T2, class... SizeT1s, class... SizeT2s>
constexpr bool operator!=(const tensor_shape<T1, SizeT1s...> &s1,
                          const tensor_shape<T2, SizeT2s...> &s2) {
  return !(s1 == s2);
}

namespace details {
template <class T, class K,
          class = std::enable_if_t<std::is_integral<K>::value>>
constexpr T _to_size_rep(const K &s) {
  return s;
}
template <class T, class K, K Val>
constexpr auto _to_size_rep(const const_ints<K, Val> &) {
  return const_ints<T, Val>();
}
}

// make_shape
namespace details {
template <class T = size_t> constexpr auto _make_shape() {
  return tensor_shape<T>();
}
template <class SizeT, class... SizeTs>
constexpr auto _make_shape(const SizeT &s, const SizeTs &... sizes) {
  using value_t = std::common_type_t<typename int_traits<SizeT>::type,
                                     typename int_traits<SizeTs>::type...>;
  return tensor_shape<value_t, decltype(details::_to_size_rep<value_t>(s)),
                      decltype(details::_to_size_rep<value_t>(sizes))...>(
      details::_to_size_rep<value_t>(s),
      details::_to_size_rep<value_t>(sizes)...);
}
}
template <class... SizeTs> constexpr auto make_shape(const SizeTs &... sizes) {
  return details::_make_shape(sizes...);
}

// cat2
namespace details {
template <class ShapeT1, size_t... I1s, class ShapeT2, size_t... I2s>
constexpr auto _cat_shape_seq(const ShapeT1 &s1, const ShapeT2 &s2,
                              const_ints<size_t, I1s...>,
                              const_ints<size_t, I2s...>) {
  return make_shape(s1.at(const_index<I1s>())..., s2.at(const_index<I2s>())...);
}
}
template <class T, class K, class... S1s, class... S2s>
constexpr auto cat2(const tensor_shape<T, S1s...> &t1,
                    const tensor_shape<K, S2s...> &t2) {
  return details::_cat_shape_seq(t1, t2,
                                 make_const_sequence(const_size_of<S1s...>()),
                                 make_const_sequence(const_size_of<S2s...>()));
}
template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const tensor_shape<T, Ss...> &a,
                    const const_ints<K, Vs...> &b) {
  return cat(a, make_shape(const_ints<K, Vs>()...));
}
template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const const_ints<K, Vs...> &a,
                    const tensor_shape<T, Ss...> &b) {
  return cat(make_shape(const_ints<K, Vs>()...), b);
}
template <class T, class... Ss, class IntT, class>
constexpr auto cat2(const tensor_shape<T, Ss...> &a, const IntT &b) {
  return cat(a, make_shape(b));
}
template <class T, class... Ss, class IntT, class>
constexpr auto cat2(const IntT &a, const tensor_shape<T, Ss...> &b) {
  return cat(make_shape(a), b);
}

// repeat_shape
namespace details {
template <class SizeT>
constexpr auto _repeat_shape(const SizeT &s, const_size<0>) {
  return tensor_shape<typename int_traits<SizeT>::type>();
}
template <class T, class... Ss>
constexpr auto _repeat_shape(const tensor_shape<T, Ss...> &s, const_size<0>) {
  return tensor_shape<T>();
}
template <class SizeT>
constexpr auto _repeat_shape(const SizeT &s, const_size<1>) {
  return make_shape(s);
}
template <class T, class... Ss>
constexpr decltype(auto) _repeat_shape(const tensor_shape<T, Ss...> &s,
                                       const_size<1>) {
  return s;
}
template <class ShapeOrSizeT, size_t Times>
constexpr auto _repeat_shape(const ShapeOrSizeT &s, const_size<Times>) {
  return cat(_repeat_shape(s, const_size<Times - 1>()), s);
}
}
template <class ShapeOrSizeT, class T, T Times>
constexpr auto repeat_shape(const ShapeOrSizeT &s,
                            const const_ints<T, Times> &times) {
  return details::_repeat_shape(s, const_size<(size_t)Times>());
}

// permute
template <class T, class... SizeTs, class... IndexTs>
constexpr auto permute(const tensor_shape<T, SizeTs...> &shape,
                       const IndexTs &... inds) {
  return make_shape(shape.at(inds)...);
}

// stream
namespace details {
template <class ShapeT, size_t... Is>
inline std::ostream &_stream_seq(std::ostream &os, const ShapeT &shape,
                                 std::index_sequence<Is...>) {
  return print_sep_to(os << "[", " ", shape.at(const_index<Is>())...) << "]";
}
}
template <class T, class... SizeTs>
inline std::ostream &operator<<(std::ostream &os,
                                const tensor_shape<T, SizeTs...> &shape) {
  return details::_stream_seq(os, shape,
                              std::make_index_sequence<sizeof...(SizeTs)>());
}
}

namespace std {

// std::get
template <size_t Idx, class T, class... SizeTs>
constexpr decltype(auto) get(const wheels::tensor_shape<T, SizeTs...> &shape) {
  return shape.at(wheels::const_index<Idx>());
}

// tuple_size
template <class T, class... SizeTs>
struct tuple_size<wheels::tensor_shape<T, SizeTs...>>
    : integral_constant<size_t, sizeof...(SizeTs)> {};
}