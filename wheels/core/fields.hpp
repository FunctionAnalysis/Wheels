#pragma once

#include <array>
#include <complex>
#include <deque>
#include <list>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "constants.hpp"
#include "iterators.hpp"
#include "overloads.hpp"
#include "types.hpp"

namespace wheels {

// func_fields
struct func_fields {
  template <class T, class U, class V>
  decltype(auto) operator()(T &&t, U &&u, V &&v) const {
    return fields(forward<T>(t), forward<U>(u), forward<V>(v));
  }
};

// fields
template <class T, class U, class V, class = std::enable_if_t<join_overloading<
                                         std::decay_t<T>, func_fields>::value>>
constexpr decltype(auto) fields(T &&t, U &&usage, V &&visitor) {
  return overloaded<func_fields,
                    category_for_overloading_t<std::decay_t<T>, func_fields>,
                    std::decay_t<U>, std::decay_t<V>>()(
      forward<T>(t), forward<U>(usage), forward<V>(visitor));
}

// fields categories
struct fields_category_tuple_like {};
struct fields_category_container {};

//// fields implementations

// empty classes -> nullptr_t
template <class T, class U, class V, class = void,
          class = std::enable_if_t<
              !join_overloading<std::decay_t<T>, func_fields>::value &&
              std::is_empty<std::decay_t<T>>::value>>
constexpr decltype(auto) fields(T &&, U &&, V &&) {
  return nullptr;
}

// tuple like types -> tuple
namespace details {
template <class TupleT, class V, size_t... Is>
auto _fields_of_tuple_seq(TupleT &&t, V &&visitor,
                          const const_ints<size_t, Is...> &) {
  return forward<V>(visitor)(std::get<Is>(forward<TupleT>(t))...);
}
}
template <class U, class V>
struct overloaded<func_fields, fields_category_tuple_like, U, V> {
  template <class TT, class UU, class VV>
  constexpr decltype(auto) operator()(TT &&t, UU &&, VV &&visitor) const {
    return details::_fields_of_tuple_seq(
        forward<TT>(t), forward<VV>(visitor),
        make_const_sequence(
            const_size<std::tuple_size<std::decay_t<TT>>::value>()));
  }
};
template <class T1, class T2>
constexpr auto category_for_overloading(const std::pair<T1, T2> &,
                                        const func_fields &) {
  return fields_category_tuple_like();
}
template <class T, size_t N>
constexpr auto category_for_overloading(const std::array<T, N> &,
                                        const func_fields &) {
  return fields_category_tuple_like();
}
template <class... Ts>
constexpr auto category_for_overloading(const std::tuple<Ts...> &,
                                        const func_fields &) {
  return fields_category_tuple_like();
}

// static raw array -> tuple
namespace details {
template <class T, size_t N, class V, size_t... Is>
auto _fields_of_raw_array_seq(T (&arr)[N], V &&visitor,
                              const const_ints<size_t, Is...> &) {
  return forward<V>(visitor)(arr[Is]...);
}
}
template <class T, size_t N, class U, class V>
constexpr decltype(auto) fields(T (&arr)[N], U &&, V &&visitor) {
  return details::_fields_of_raw_array_seq(
      arr, forward<V>(visitor), make_const_sequence(const_size<N>()));
}

// container types -> container_proxy
template <class ContT, class VisitorT> class container_proxy {
public:
  template <class C, class V>
  constexpr container_proxy(C &&c, V &&v)
      : _content(forward<C>(c)), _visitor(forward<V>(v)) {}
  constexpr container_proxy(const container_proxy &) = default;
  container_proxy(container_proxy &&) = default;
  container_proxy &operator=(const container_proxy &) = default;
  container_proxy &operator=(container_proxy &&) = default;
  template <class ContT2>
  container_proxy &operator=(const container_proxy<ContT2, VisitorT> &c) {
    _content = ContT(std::begin(c._content), std::end(c._content));
    _visitor = c._visitor;
    return *this;
  }

  const ContT &content() const { return _content; }
  ContT &content() { return _content; }
  const VisitorT &visitor() const { return _visitor; }
  VisitorT &visitor() { return _visitor; }

  decltype(auto) begin() const {
    return make_transform_iterator(std::begin(_content), _visit_functor());
  }
  decltype(auto) end() const {
    return make_transform_iterator(std::end(_content), _visit_functor());
  }
  decltype(auto) begin() {
    return make_transform_iterator(std::begin(_content), _visit_functor());
  }
  decltype(auto) end() {
    return make_transform_iterator(std::end(_content), _visit_functor());
  }

  auto size() const { return _content.size(); }
  decltype(auto) operator[](size_t i) const {
    return _visitor.visit(*std::next(std::begin(_content), i));
  }
  decltype(auto) operator[](size_t i) {
    return _visitor.visit(*std::next(std::begin(_content), i));
  }

private:
  auto _visit_functor() {
    return [this](auto &&e) { return _visitor.visit(e); };
  }
  auto _visit_functor() const {
    return [this](auto &&e) { return _visitor.visit(e); };
  }

private:
  ContT _content;
  VisitorT _visitor;
};

template <class ContT1, class ContT2, class V>
bool operator==(const container_proxy<ContT1, V> &c1,
                const container_proxy<ContT2, V> &c2) {
  return std::equal(c1.begin(), c1.end(), c2.begin(), c2.end());
}
template <class ContT1, class ContT2, class V>
constexpr bool operator!=(const container_proxy<ContT1, V> &c1,
                          const container_proxy<ContT2, V> &c2) {
  return !(c1 == c2);
}
template <class ContT1, class ContT2, class V>
bool operator<(const container_proxy<ContT1, V> &c1,
               const container_proxy<ContT2, V> &c2) {
  return std::lexicographical_compare(c1.begin(), c1.end(), c2.begin(),
                                      c2.end());
}
template <class ContT1, class ContT2, class V>
bool operator>(const container_proxy<ContT1, V> &c1,
               const container_proxy<ContT2, V> &c2) {
  return c2 < c1;
}
template <class ContT1, class ContT2, class V>
bool operator<=(const container_proxy<ContT1, V> &c1,
                const container_proxy<ContT2, V> &c2) {
  return !(c1 > c2);
}
template <class ContT1, class ContT2, class V>
bool operator>=(const container_proxy<ContT1, V> &c1,
                const container_proxy<ContT2, V> &c2) {
  return !(c1 < c2);
}

template <class T> struct is_container_proxy : no {};
template <class T, class V>
struct is_container_proxy<container_proxy<T, V>> : yes {};

// as_container
template <class ContT, class VisitorT>
constexpr auto as_container(ContT &&c, VisitorT &&v) {
  return container_proxy<ContT, VisitorT>(forward<ContT>(c),
                                          forward<VisitorT>(v));
}

template <class U, class V>
struct overloaded<func_fields, fields_category_container, U, V> {
  template <class TT, class UU, class VV>
  constexpr decltype(auto) operator()(TT &&t, UU &&, VV &v) const {
    return v(as_container(forward<TT>(t), v));
  }
};

template <class T, class AllocT>
constexpr auto category_for_overloading(const std::vector<T, AllocT> &,
                                        const func_fields &) {
  return fields_category_container();
};
template <class T, class AllocT>
constexpr auto category_for_overloading(const std::list<T, AllocT> &,
                                        const func_fields &) {
  return fields_category_container();
};
template <class T, class AllocT>
constexpr auto category_for_overloading(const std::deque<T, AllocT> &,
                                        const func_fields &) {
  return fields_category_container();
};

// has_member_func_fields
struct test_visitor {
  template <class... Ts> nullptr_t operator()(Ts &&...) { return nullptr; }
};
namespace details {
template <class T, class UsageT> struct _has_member_func_fields {
  template <class TT, class UU>
  static auto test(int)
      -> decltype(std::declval<TT>().fields(std::declval<UU>(), test_visitor()),
                  yes()) {
    return yes();
  }
  template <class, class> static no test(...) { return no(); }
  static const bool value =
      std::is_same<decltype(test<T, UsageT>(1)), yes>::value;
};
}
template <class T, class UsageT>
struct has_member_func_fields
    : const_bool<details::_has_member_func_fields<T, UsageT>::value> {};

// has_member_func_fields_simple
namespace details {
template <class T> struct _has_member_func_fields_simple {
  template <class TT>
  static auto test(int)
      -> decltype(std::declval<TT>().fields(test_visitor()), yes()) {
    return yes();
  }
  template <class> static no test(...) { return no(); }
  static const bool value = std::is_same<decltype(test<T>(1)), yes>::value;
};
}
template <class T>
struct has_member_func_fields_simple
    : const_bool<details::_has_member_func_fields_simple<T>::value> {};

// has_global_func_fields
namespace details {
template <class T, class UsageT> struct _has_global_func_fields {
  template <class TT, class UU>
  static auto test(int)
      -> decltype(::wheels::fields(std::declval<TT>(), std::declval<UU>(),
                                   test_visitor()),
                  yes()) {
    return yes();
  }
  template <class, class> static no test(...) { return no(); }
  static const bool value =
      std::is_same<decltype(test<T, UsageT>(1)), yes>::value;
};
}
template <class T, class UsageT>
struct has_global_func_fields
    : const_bool<details::_has_global_func_fields<T, UsageT>::value> {};

// has_global_func_fields_simple
namespace details {
template <class T> struct _has_global_func_fields_simple {
  template <class TT>
  static auto test(int)
      -> decltype(::wheels::fields(std::declval<TT>(), test_visitor()), yes()) {
    return yes();
  }
  template <class> static no test(...) { return no(); }
  static const bool value = std::is_same<decltype(test<T>(1)), yes>::value;
};
}
template <class T>
struct has_global_func_fields_simple
    : const_bool<details::_has_global_func_fields_simple<T>::value> {};

// field_visitor
template <class PackT, class ProcessT, class UsageT> class field_visitor {
  using this_t = field_visitor<PackT, ProcessT, UsageT>;

  this_t &non_const_self() const { return const_cast<this_t &>(*this); }
  this_t &non_const_self() { *this; }
  const this_t &const_self() { return *this; }
  constexpr const this_t &const_self() const { return *this; }

public:
  template <class PP, class RR>
  constexpr field_visitor(PP &&p, RR &&r)
      : _pack(forward<PP>(p)), _process(forward<RR>(r)) {}

// visit single member
#define WHEELS_PARAMETER_DISTINGUISH(i) const_size<i> = const_size<i>()

  // has_member_func_fields<T...>
  template <class T,
            class = std::enable_if_t<has_member_func_fields<T, UsageT>::value>>
  constexpr decltype(auto) visit(T &&v, WHEELS_PARAMETER_DISTINGUISH(0)) const {
    return forward<T>(v).fields(UsageT(), non_const_self());
  }
  // has_member_func_fields<T &...> for const T
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<const T &, UsageT>::value &&
                         has_member_func_fields<T &, UsageT>::value>>
  constexpr decltype(auto) visit(const T &v,
                                 WHEELS_PARAMETER_DISTINGUISH(0)) const {
    return const_cast<T &>(v).fields(UsageT(), const_self());
  }

  // has_member_func_fields_simple
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<T, UsageT>::value &&
                         has_member_func_fields_simple<T>::value>>
  constexpr decltype(auto) visit(T &&v, WHEELS_PARAMETER_DISTINGUISH(1)) const {
    return forward<T>(v).fields(non_const_self());
  }
  // has_member_func_fields_simple<T &...> for const T
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<const T &, UsageT>::value &&
                         !has_member_func_fields<T &, UsageT>::value &&
                         !has_member_func_fields_simple<const T &>::value &&
                         has_member_func_fields_simple<T &>::value>>
  constexpr decltype(auto) visit(const T &v,
                                 WHEELS_PARAMETER_DISTINGUISH(1)) const {
    return const_cast<T &>(v).fields(const_self());
  }

  // has_global_func_fields
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<T, UsageT>::value &&
                         !has_member_func_fields_simple<T>::value &&
                         has_global_func_fields<T, UsageT>::value>>
  constexpr decltype(auto) visit(T &&v, WHEELS_PARAMETER_DISTINGUISH(2)) const {
    return ::wheels::fields(forward<T>(v), UsageT(), non_const_self());
  }
  // has_global_func_fields<T&...> for const T
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<const T &, UsageT>::value &&
                         !has_member_func_fields<T &, UsageT>::value &&
                         !has_member_func_fields_simple<const T &>::value &&
                         !has_member_func_fields_simple<T &>::value &&
                         !has_global_func_fields<const T &, UsageT>::value &&
                         has_global_func_fields<T &, UsageT>::value>>
  constexpr decltype(auto) visit(const T &v,
                                 WHEELS_PARAMETER_DISTINGUISH(2)) const {
    return ::wheels::fields(const_cast<T &>(v), UsageT(), const_self());
  }

  // has_global_func_fields_simple
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<T, UsageT>::value &&
                         !has_member_func_fields_simple<T>::value &&
                         !has_global_func_fields<T, UsageT>::value &&
                         has_global_func_fields_simple<T>::value>>
  constexpr decltype(auto) visit(T &&v, WHEELS_PARAMETER_DISTINGUISH(3)) const {
    return ::wheels::fields(forward<T>(v), non_const_self());
  }
  // has_global_func_fields_simple<T &...> for const T
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<const T &, UsageT>::value &&
                         !has_member_func_fields<T &, UsageT>::value &&
                         !has_member_func_fields_simple<const T &>::value &&
                         !has_member_func_fields_simple<T &>::value &&
                         !has_global_func_fields<const T &, UsageT>::value &&
                         !has_global_func_fields<T &, UsageT>::value &&
                         !has_global_func_fields_simple<const T &>::value &&
                         has_global_func_fields_simple<T &>::value>>
  constexpr decltype(auto) visit(const T &v,
                                 WHEELS_PARAMETER_DISTINGUISH(3)) const {
    return ::wheels::fields(const_cast<T &>(v), const_self());
  }

  // leaf type or container type
  template <class T, class = std::enable_if_t<
                         !has_member_func_fields<T, UsageT>::value &&
                         !has_member_func_fields_simple<T>::value &&
                         !has_global_func_fields<T, UsageT>::value &&
                         !has_global_func_fields_simple<T>::value>>
  constexpr decltype(auto) visit(T &&v, WHEELS_PARAMETER_DISTINGUISH(4)) const {
    return _process(forward<T>(v));
  }

#undef WHEELS_PARAMETER_DISTINGUISH

  // pack all members
  template <class... Ts> decltype(auto) operator()(Ts &&... vs) {
    return _pack(visit(forward<Ts>(vs))...);
  }
  template <class... Ts>
  constexpr decltype(auto)
  operator()(const Ts &... vs) const { // const visitor(non const fields ...) ->
                                       // visitor(const fields ...)
    return _pack(visit(vs)...);
  }

private:
  PackT _pack;
  ProcessT _process;
};

// make_field_visitor
template <class UU, class PP, class RR>
constexpr auto make_field_visitor(PP &&pack, RR &&proc) {
  return field_visitor<std::decay_t<PP>, std::decay_t<RR>, UU>(
      forward<PP>(pack), forward<RR>(proc));
}

struct visit_to_tuplize {};
struct pack_as_tuple {
  template <class ContT, class VisitorT>
  constexpr auto operator()(container_proxy<ContT, VisitorT> &&c) const {
    return std::move(c);
  }
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... args) const {
    return std::tuple<ArgTs...>(forward<ArgTs>(args)...);
  }
};
struct process_direct_pass {
  template <class ArgT> constexpr ArgT &&operator()(ArgT &&arg) const {
    return static_cast<ArgT &&>(arg);
  }
};

// tuplize
template <class T> constexpr auto tuplize(T &&data) {
  return make_field_visitor<visit_to_tuplize>(pack_as_tuple(),
                                              process_direct_pass())
      .visit(forward<T>(data));
}

// traverse_fields
struct visit_to_traverse {};
struct pack_nothing {
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return nullptr;
  }
};
template <class FunT> struct process_by_traverse {
  constexpr process_by_traverse(FunT f) : fun(f) {}
  template <class ContT, class VisitorT>
  auto operator()(const container_proxy<ContT, VisitorT> &c) const {
    for (auto it = c.begin(); it != c.end(); ++it) {
      *it;
    }
    return nullptr;
  }
  template <class ContT, class VisitorT>
  auto operator()(container_proxy<ContT, VisitorT> &c) const {
    for (auto it = c.begin(); it != c.end(); ++it) {
      *it;
    }
    return nullptr;
  }
  template <class ContT, class VisitorT>
  auto operator()(container_proxy<ContT, VisitorT> &&c) const {
    for (auto it = c.begin(); it != c.end(); ++it) {
      *it;
    }
    return nullptr;
  }
  template <class ArgT> auto operator()(ArgT &&arg) const {
    fun(forward<ArgT>(arg));
    return nullptr;
  }
  FunT fun;
};
template <class T, class FunT>
constexpr void traverse_fields(T &&data, FunT fun) {
  make_field_visitor<visit_to_traverse>(pack_nothing(),
                                        process_by_traverse<FunT>(fun))
      .visit(forward<T>(data));
}

// randomize_fields
template <class RNG> struct randomizer {
  RNG &rng;
  constexpr randomizer(RNG &r) : rng(r) {}

  template <class T>
  std::enable_if_t<std::is_integral<T>::value> operator()(T &v) const {
    const T minv = std::numeric_limits<T>::lowest();
    const T maxv = std::numeric_limits<T>::max();
    v = static_cast<T>(std::uniform_int_distribution<std::intmax_t>(
        (std::intmax_t)minv, (std::intmax_t)maxv)(rng));
  }
  template <class T>
  std::enable_if_t<std::is_integral<T>::value>
  operator()(std::complex<T> &v) const {
    const T minv = std::numeric_limits<T>::lowest();
    const T maxv = std::numeric_limits<T>::max();
    std::uniform_int_distribution<std::intmax_t> dist((std::intmax_t)minv,
                                                      (std::intmax_t)maxv);
    v.real(static_cast<T>(dist(rng)));
    v.imag(static_cast<T>(dist(rng)));
  }
  template <class T, class = void>
  std::enable_if_t<std::is_floating_point<T>::value> operator()(T &v) const {
    const T minv = T(-1.0);
    const T maxv = T(1.0);
    v = std::uniform_real_distribution<T>(minv, maxv)(rng);
  }
  template <class T, class = void>
  std::enable_if_t<std::is_floating_point<T>::value>
  operator()(std::complex<T> &v) const {
    const T minv = T(-1.0);
    const T maxv = T(1.0);
    std::uniform_real_distribution<T> dist(minv, maxv);
    v.real(dist(rng));
    v.imag(dist(rng));
  }
};
template <class T, class RNG> inline void randomize_fields(T &data, RNG &rng) {
  traverse_fields(data, randomizer<RNG>(rng));
}

// any_of_fields
struct pack_by_any {
  template <class... ArgTs> constexpr bool operator()(ArgTs &&... args) const {
    return any(forward<ArgTs>(args)...);
  }
};
template <class CheckFunT> struct process_by_any {
  template <class ContT, class VisitorT>
  constexpr bool operator()(const container_proxy<ContT, VisitorT> &c) const {
    return std::any_of(c.begin(), c.end(), [](auto &&e) { return !!e; });
  }
  template <class ArgT> constexpr bool operator()(const ArgT &arg) const {
    return checker(arg);
  }
  CheckFunT checker;
};
template <class T, class CheckFunT>
constexpr bool any_of_fields(const T &data, CheckFunT checker) {
  return make_field_visitor<visit_to_traverse>(
             pack_by_any(), process_by_any<CheckFunT>{checker})
      .visit(data);
}

// none_of_fields
template <class T, class CheckFunT>
constexpr bool none_of_fields(const T &data, CheckFunT checker) {
  return !any_of_fields(data, checker);
}

// all_of_fields
struct pack_by_all {
  template <class... ArgTs> constexpr bool operator()(ArgTs &&... args) const {
    return all(forward<ArgTs>(args)...);
  }
};
template <class CheckFunT> struct process_by_all {
  template <class ContT, class VisitorT>
  constexpr bool operator()(const container_proxy<ContT, VisitorT> &c) const {
    return std::all_of(c.begin(), c.end(), [](auto &&e) { return e; });
  }
  template <class ArgT> constexpr bool operator()(const ArgT &arg) const {
    return checker(arg);
  }
  CheckFunT checker;
};
template <class T, class CheckFunT>
constexpr bool all_of_fields(const T &data, CheckFunT checker) {
  return make_field_visitor<visit_to_traverse>(
             pack_by_all(), process_by_all<CheckFunT>{checker})
      .visit(data);
}

// comparable
template <class T, class Kind = T> struct comparable {
  constexpr decltype(auto) as_tuple() const {
    using result_t = decltype(tuplize(static_cast<const T &>(*this)));
    static_assert(!std::is_same<T, result_t>::value, "tuplization failed");
    return tuplize(static_cast<const T &>(*this));
  }
};

template <class A, class B, class Kind>
constexpr bool operator==(const comparable<A, Kind> &a,
                          const comparable<B, Kind> &b) {
  return a.as_tuple() == b.as_tuple();
}
template <class A, class B, class Kind>
constexpr bool operator!=(const comparable<A, Kind> &a,
                          const comparable<B, Kind> &b) {
  return a.as_tuple() != b.as_tuple();
}
template <class A, class B, class Kind>
constexpr bool operator<(const comparable<A, Kind> &a,
                         const comparable<B, Kind> &b) {
  return a.as_tuple() < b.as_tuple();
}
template <class A, class B, class Kind>
constexpr bool operator<=(const comparable<A, Kind> &a,
                          const comparable<B, Kind> &b) {
  return a.as_tuple() <= b.as_tuple();
}
template <class A, class B, class Kind>
constexpr bool operator>(const comparable<A, Kind> &a,
                         const comparable<B, Kind> &b) {
  return a.as_tuple() > b.as_tuple();
}
template <class A, class B, class Kind>
constexpr bool operator>=(const comparable<A, Kind> &a,
                          const comparable<B, Kind> &b) {
  return a.as_tuple() >= b.as_tuple();
}

// convertible
template <class T, class Kind = void> struct convertible {
  template <class K> T &operator=(const convertible<K, Kind> &c) {
    if (this != reinterpret_cast<const convertible<T, Kind> *>(&c)) {
      tuplize(static_cast<T &>(*this)) = tuplize(static_cast<const K &>(c));
    }
    return static_cast<T &>(*this);
  }
};
}
