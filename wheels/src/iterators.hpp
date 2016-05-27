#pragma once

#include <iterator>

#include "utility_fwd.hpp"

#include "const_ints.hpp"

namespace wheels {

// is_iterator
template <class T, class = void> struct is_iterator : no {};
template <class T>
struct is_iterator<
    T, void_t<typename T::iterator_category, typename T::value_type,
                   typename T::difference_type, typename T::pointer,
                   typename T::reference>> : yes {};
template <class T> struct is_iterator<T *> : yes {};

// transform_iterator
template <class T, class IterT, class FunT>
class transform_iterator
    : public std::iterator<
          typename std::iterator_traits<IterT>::iterator_category, T,
          typename std::iterator_traits<IterT>::difference_type> {
  static_assert(is_iterator<IterT>::value, "IterT must be an iterator type");

public:
  using difference_type = typename std::iterator_traits<IterT>::difference_type;

  constexpr transform_iterator() : current(), fun() {}
  template <class F>
  constexpr transform_iterator(IterT it, F &&f)
      : current(it), fun(std::forward<F>(f)) {}
  constexpr IterT base() const { return current; }

  decltype(auto) operator*() const { return fun(*current); }
  decltype(auto) operator-> () const { return &(**this); }

  transform_iterator &operator++() {
    ++current;
    return *this;
  }
  transform_iterator &operator++(int) {
    auto tmp = *this;
    ++current;
    return tmp;
  }
  transform_iterator &operator--() {
    --current;
    return *this;
  }
  transform_iterator &operator--(int) {
    auto tmp = *this;
    --current;
    return tmp;
  }

  transform_iterator &operator+=(difference_type off) { // increment by integer
    current += off;
    return (*this);
  }
  transform_iterator
  operator+(difference_type off) const { // return this + integer
    return (transform_iterator(current + off, fun));
  }

  transform_iterator &operator-=(difference_type off) { // decrement by integer
    current -= off;
    return (*this);
  }
  transform_iterator
  operator-(difference_type off) const { // return this - integer
    return (transform_iterator(current - off, fun));
  }

protected:
  IterT current;
  FunT fun;
};

template <class T, class IterT, class FunT>
constexpr bool operator==(const transform_iterator<T, IterT, FunT> &i1,
                          const transform_iterator<T, IterT, FunT> &i2) {
  return i1.base() == i2.base();
}
template <class T, class IterT, class FunT>
constexpr bool operator!=(const transform_iterator<T, IterT, FunT> &i1,
                          const transform_iterator<T, IterT, FunT> &i2) {
  return !(i1 == i2);
}
template <class T, class IterT, class FunT>
constexpr bool operator<(const transform_iterator<T, IterT, FunT> &i1,
                         const transform_iterator<T, IterT, FunT> &i2) {
  return i1.base() < i2.base();
}

template <class T, class IterT, class FunT>
constexpr auto operator-(const transform_iterator<T, IterT, FunT> &i1,
                         const transform_iterator<T, IterT, FunT> &i2) {
  return i1.base() - i2.base();
}

// make_transform_iterator
template <class IterT, class FunT>
constexpr auto make_transform_iterator(IterT it, FunT f) {
  using value_t = std::decay_t<decltype(f(*it))>;
  return transform_iterator<value_t, IterT, FunT>(it, f);
}

// interval
template <class FromT, class ToT> class interval {
public:
  constexpr interval(const FromT &b, const ToT &e) : _begin(b), _end(e) {}
  constexpr FromT begin() const { return _begin; }
  constexpr ToT end() const { return _end; }
  constexpr auto size() const { return _end - _begin; }

private:
  FromT _begin;
  ToT _end;
};

// make_interval
template <class FromT, class ToT>
constexpr auto make_interval(const FromT &begin, const ToT &end) {
  return interval<FromT, ToT>(begin, end);
}
// span
template <class FromT, class SizeT = const_size<1>>
constexpr auto span(const FromT &begin, const SizeT &s = SizeT()) {
  return make_interval(begin, begin + s);
}

// is_interval
template <class T> struct is_interval : no {};
template <class FromT, class ToT>
struct is_interval<interval<FromT, ToT>> : yes {};
}
