#pragma once

#include "aligned.hpp"
#include "ewise.hpp"
#include "storage.hpp"
#include "tensor_view_base.hpp"

#include "tensor_fwd.hpp"

namespace wheels {

namespace details {
template <class ShapeT, size_t... Is>
constexpr std::enable_if_t<ShapeT::dynamic_size_num == 1, ShapeT>
_make_shape_from_magnitude_seq(size_t magnitude, const_ints<size_t, Is...>) {
  static_assert(ShapeT::last_dynamic_dim >= 0,
                "ShapeT::last_dynamic_dim is not valid");
  return ShapeT(conditional(const_bool<Is == ShapeT::last_dynamic_dim>(),
                            magnitude / ShapeT::static_magnitude,
                            std::ignore)...);
}
template <class ShapeT, size_t... Is>
constexpr std::enable_if_t<ShapeT::dynamic_size_num == 0, ShapeT>
_make_shape_from_magnitude_seq(size_t magnitude, const_ints<size_t, Is...>) {
  return ShapeT();
}
}

// tensor
template <class ET, class ShapeT>
class tensor : public tensor_view_base<ET, ShapeT, tensor<ET, ShapeT>, true> {
  using _base_t = tensor_view_base<ET, ShapeT, tensor<ET, ShapeT>, true>;

public:
  using value_type = ET;
  using shape_type = ShapeT;

  // tensor()
  constexpr tensor() : _storage() {}

  // tensor(shape)
  constexpr explicit tensor(const ShapeT &shape) : _storage(shape) {}

  // tensor(shape, e)
  constexpr tensor(const ShapeT &shape, const value_type &v)
      : _storage(shape, v) {}

  // tensor(shape, with_elements, e1, e2, e3 ...)
  template <class... EleTs>
  constexpr tensor(const ShapeT &shape, _with_elements we, EleTs &&... eles)
      : _storage(shape, we, std::forward<EleTs>(eles)...) {}

  // tensor({e1, e2, e3 ...})
  template <class EleT>
  constexpr tensor(std::initializer_list<EleT> ilist)
      : _storage(
            details::_make_shape_from_magnitude_seq<ShapeT>(
                ilist.size(), make_const_sequence(const_size<ShapeT::rank>())),
            with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(shape, {e1, e2, e3 ...})
  constexpr tensor(const ShapeT &shape, std::initializer_list<value_type> ilist)
      : _storage(shape, with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(begin, end)
  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(IterT begin, IterT end)
      : _storage(details::_make_shape_from_magnitude_seq<ShapeT>(
                     std::distance(begin, end),
                     make_const_sequence(const_size<ShapeT::rank>())),
                 with_iterators, begin, end) {}

  // tensor(shape, begin, end)
  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(const ShapeT &shape, IterT begin, IterT end)
      : _storage(shape, with_iterators, begin, end) {}

  // tensor(e1, e2, e3, ...) for statically shaped tensors
  template <class FirstEleT, class... EleTs,
            bool B = (ShapeT::static_magnitude == 1 + sizeof...(EleTs) &&
                      std::is_convertible<FirstEleT, ET>::value),
            class = std::enable_if_t<B>>
  constexpr explicit tensor(FirstEleT &&firstE, EleTs &&... eles)
      : _storage(ShapeT(), with_elements, std::forward<FirstEleT>(firstE),
                 std::forward<EleTs>(eles)...) {}

  tensor(const tensor &) = default;
  tensor(tensor &&) = default;
  tensor &operator=(const tensor &) = default;
  tensor &operator=(tensor &&) = default;

  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  constexpr tensor(const tensor_base<ET, AnotherShapeT, AnotherT> &another)
      : _storage(another.shape()) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<!std::is_same<ET, AnotherET>::value &&
                                     AnotherShapeT::rank == ShapeT::rank>>
  constexpr explicit tensor(
      const tensor_base<AnotherET, AnotherShapeT, AnotherT> &another)
      : _storage(another.shape()) {
    assign_elements_forced(*this, another.derived());
  }

  constexpr const ET *ptr() const { return _storage.data(); }
  ET *ptr() { return _storage.data(); }
  constexpr decltype(auto) shape() const { return _storage.shape(); }
  void reshape(const shape_type &s) { _storage.reshape(s); }

  template <class ArcT> void serialize(ArcT &ar) { ar(_storage); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_storage);
  }

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

private:
  storage<value_type, shape_type> _storage;
};

// shape_of
template <class ET, class ShapeT>
constexpr decltype(auto) shape_of(const tensor<ET, ShapeT> &t) {
  return t.shape();
}

// ptr_of
template <class ET, class ShapeT>
constexpr const ET *ptr_of(const tensor<ET, ShapeT> &t) {
  return t.ptr();
}
template <class ET, class ShapeT> ET *ptr_of(tensor<ET, ShapeT> &t) {
  return t.ptr();
}

// reserve_shape
template <class ET, class ShapeT, class ST, class... SizeTs>
void reserve_shape(tensor<ET, ShapeT> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  t.reshape(shape);
}

// randomize
template <class ET, class ST, class... SizeTs, class RNG>
inline void randomize(tensor<ET, tensor_shape<ST, SizeTs...>> &ts, RNG &rng) {
  for (auto &e : ts) {
    randomize(e, rng);
  }
}

// rand
template <class ET, class ST, class... SizeTs, class RNG>
inline tensor<ET, tensor_shape<ST, SizeTs...>>
rand(const tensor_shape<ST, SizeTs...> &shape, RNG &rng) {
  tensor<ET, tensor_shape<ST, SizeTs...>> ts(shape);
  for (auto &e : ts) {
    randomize(e, rng);
  }
  return ts;
}
}