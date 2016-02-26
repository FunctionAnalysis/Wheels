#pragma once

#include "const_ints.hpp"
#include "utility.hpp"

namespace wheels {

// combination_config
template <class T, bool Stored> struct combination_config {};

// combination
template <class... T> class combination;
template <> class combination<> {
public:
  constexpr combination() {}
  static constexpr size_t size = 0;
};

template <class T, class... RestTs>
class combination<combination_config<T, true>, RestTs...>
    : public combination<RestTs...> {
  using _rest_t = combination<RestTs...>;

public:
  constexpr const _rest_t &rest() const {
    return static_cast<const _rest_t &>(*this);
  }
  _rest_t &rest() { return static_cast<_rest_t &>(*this); }

public:
  template <class TT, class... RestTTs>
  constexpr combination(TT &&v, RestTTs &&... restVs)
      : _val(forward<TT>(v)), _rest_t(forward<RestTTs>(restVs)...) {}
  constexpr combination() : _val(), _rest_t() {}

public:
  static constexpr size_t size = 1 + sizeof...(RestTs);

  constexpr const T &value() const { return _val; }
  T &value() { return _val; }

  constexpr const T &at(const const_index<0> &) const { return value(); }
  T &at(const const_index<0> &) { return value(); }

  template <size_t Idx>
  constexpr decltype(auto) at(const const_index<Idx> &) const {
    static_assert(Idx < size, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }
  template <size_t Idx> decltype(auto) at(const const_index<Idx> &) {
    static_assert(Idx < size, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }
  template <class K, K Idx, bool _B = std::is_same<K, size_t>::value,
            wheels_enable_if(!_B)>
  constexpr decltype(auto) at(const const_ints<K, Idx> &) const {
    static_assert(Idx < size, "Idx too large");
    return at(const_index<Idx>());
  }
  template <class K, K Idx, bool _B = std::is_same<K, size_t>::value,
            wheels_enable_if(!_B)>
  decltype(auto) at(const const_ints<K, Idx> &) {
    static_assert(Idx < size, "Idx too large");
    return at(const_index<Idx>());
  }

  template <class K, K Idx>
  constexpr decltype(auto) operator[](const const_ints<K, Idx> &i) const {
    static_assert(Idx < size, "Idx too large");
    return at(i);
  }
  template <class K, K Idx>
  decltype(auto) operator[](const const_ints<K, Idx> &i) {
    static_assert(Idx < size, "Idx too large");
    return at(i);
  }

private:
  T _val;
};

template <class T, class... RestTs>
class combination<combination_config<T, false>, RestTs...>
    : public combination<RestTs...> {
  using _rest_t = combination<RestTs...>;

public:
  constexpr const _rest_t &rest() const {
    return static_cast<const _rest_t &>(*this);
  }
  _rest_t &rest() { return static_cast<_rest_t &>(*this); }

public:
  template <class TT, class... RestTTs>
  constexpr combination(TT &&v, RestTTs &&... restVs)
      : _rest_t(forward<RestTTs>(restVs)...) {}
  template <class... RestTTs>
  constexpr combination(ignore_t, RestTTs &&... restVs)
      : _rest_t(forward<RestTTs>(restVs)...) {}
  constexpr combination() : _rest_t() {}

public:
  static constexpr size_t size = 1 + sizeof...(RestTs);

  constexpr T value() const { return T(); }

  constexpr T at(const const_index<0> &) const { return value(); }
  T at(const const_index<0> &) { return value(); }

  template <size_t Idx>
  constexpr decltype(auto) at(const const_index<Idx> &) const {
    static_assert(Idx < size, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }
  template <size_t Idx> decltype(auto) at(const const_index<Idx> &) {
    static_assert(Idx < size, "Idx too large");
    return rest().at(const_index<Idx - 1>());
  }

  template <class K, K Idx, bool _B = std::is_same<K, size_t>::value,
            wheels_enable_if(!_B)>
  constexpr decltype(auto) at(const const_ints<K, Idx> &) const {
    static_assert(Idx < size, "Idx too large");
    return at(const_index<Idx>());
  }
  template <class K, K Idx, bool _B = std::is_same<K, size_t>::value,
            wheels_enable_if(!_B)>
  decltype(auto) at(const const_ints<K, Idx> &) {
    static_assert(Idx < size, "Idx too large");
    return at(const_index<Idx>());
  }

  template <class K, K Idx>
  constexpr decltype(auto) operator[](const const_ints<K, Idx> &i) const {
    static_assert(Idx < size, "Idx too large");
    return at(i);
  }
  template <class K, K Idx>
  decltype(auto) operator[](const const_ints<K, Idx> &i) {
    static_assert(Idx < size, "Idx too large");
    return at(i);
  }
};

// is_combination
template <class T> struct is_combination : no {};
template <class... Ts> struct is_combination<combination<Ts...>> : yes {};

// combine
template <class... Ts> constexpr auto combine(Ts &&... ts) {
  return combination<combination_config<Ts, !(std::is_class<Ts>::value &&
                                              std::is_empty<Ts>::value)>...>(
      forward<Ts>(ts)...);
}

// copy_all
template <class... Ts> constexpr auto copy_all(Ts &&... ts) {
  return combination<combination_config<
      std::decay_t<Ts>, !(std::is_class<std::decay_t<Ts>>::value &&
                          std::is_empty<std::decay_t<Ts>>::value)>...>(
      forward<Ts>(ts)...);
}
}

namespace std {

// std::get
template <size_t Idx, class T, class... RestTs>
constexpr decltype(auto) get(const wheels::combination<T, RestTs...> &c) {
  return c.at(wheels::const_index<Idx>());
}
template <size_t Idx, class T, class... RestTs>
decltype(auto) get(wheels::combination<T, RestTs...> &c) {
  return c.at(wheels::const_index<Idx>());
}

// tuple_size
template <class... Ts>
struct tuple_size<wheels::combination<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)> {};
}
