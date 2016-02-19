#pragma once

#include "aligned.hpp"
#include "ewise_ops.hpp"
#include "storage.hpp"

namespace wheels {

namespace details {
template <class ShapeT, size_t... Is>
constexpr ShapeT _make_shape_from_magnitude_seq(size_t magnitude,
                                                const_ints<size_t, Is...>) {
  static_assert(ShapeT::dynamic_size_num == 1,
                "ShapeT::dynamic_size_num should be 1 here");
  static_assert(ShapeT::last_dynamic_dim >= 0,
                "ShapeT::last_dynamic_dim is not valid");
  return ShapeT(conditional(const_bool<Is == ShapeT::last_dynamic_dim>(),
                            magnitude / ShapeT::static_magnitude,
                            std::ignore)...);
}
}

// tensor
template <class ET, class ShapeT>
class tensor
    : public tensor_continuous_data_base<ET, ShapeT, tensor<ET, ShapeT>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;

  // tensor()
  constexpr tensor() : _storage() {}

  // tensor(shape)
  constexpr tensor(const ShapeT &shape) : _storage(shape) {}

  // tensor(shape, e)
  constexpr tensor(const ShapeT &shape, const value_type &v)
      : _storage(shape, v) {}

  // tensor(shape, with_elements, e1, e2, e3 ...)
  template <class... EleTs>
  constexpr tensor(const ShapeT &shape, _with_elements we, EleTs &&... eles)
      : _storage(shape, we, forward<EleTs>(eles)...) {}

  // tensor({e1, e2, e3 ...})
  template <class EleT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0)>>
  constexpr tensor(std::initializer_list<EleT> ilist)
      : _storage(ShapeT(), with_iterators, ilist.begin(), ilist.end()) {}
  template <class EleT, class = void,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1)>>
  constexpr tensor(std::initializer_list<EleT> ilist)
      : _storage(
            details::_make_shape_from_magnitude_seq<ShapeT>(
                ilist.size(), make_const_sequence(const_size<ShapeT::rank>())),
            with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(shape, {e1, e2, e3 ...})
  constexpr tensor(const ShapeT &shape, std::initializer_list<value_type> ilist)
      : _storage(shape, with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(begin, end)
  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, yes * = nullptr)
      : _storage(ShapeT(), with_iterators, begin, end) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, no * = nullptr)
      : _storage(details::_make_shape_from_magnitude_seq<ShapeT>(
                     std::distance(begin, end),
                     make_const_sequence(const_size<ShapeT::rank>())),
                 with_iterators, begin, end) {}

  // tensor(shape, begin, end)
  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(const ShapeT &shape, IterT begin, IterT end)
      : _storage(shape, with_iterators, begin, end) {}

  // tensor(e1, e2, e3 ...)
  template <class... EleTs,
            bool B = (ShapeT::dynamic_size_num == 0 &&
                      ::wheels::all(std::is_convertible<EleTs, ET>::value...)),
            class = std::enable_if_t<B>>
  constexpr tensor(EleTs &&... eles)
      : _storage(ShapeT(), with_elements, forward<EleTs>(eles)...) {}

  template <class... EleTs, class = void,
            bool B = (ShapeT::dynamic_size_num == 1 &&
                      ::wheels::all(std::is_convertible<EleTs, ET>::value...)),
            class = std::enable_if_t<B>>
  constexpr tensor(EleTs &&... eles)
      : _storage(details::_make_shape_from_magnitude_seq<ShapeT>(
                     sizeof...(EleTs),
                     make_const_sequence(const_size<ShapeT::rank>())),
                 with_elements, forward<EleTs>(eles)...) {}

  tensor(const tensor &) = default;
  tensor(tensor &&) = default;
  tensor &operator=(const tensor &) = default;
  tensor &operator=(tensor &&) = default;

  template <class AnotherT>
  constexpr tensor(const tensor_core<AnotherT> &another)
      : _storage(another.shape()) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherT>
  tensor &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

  constexpr const ET *ptr() const { return _storage.data(); }
  ET *ptr() { return _storage.data(); }
  constexpr decltype(auto) shape() const { return _storage.shape(); }
  void reshape(const shape_type &s) { _storage.reshape(s); }

  template <class ArcT> void serialize(ArcT &ar) { ar(_storage); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_storage);
  }

  // +=
  template <class T> tensor &operator+=(const tensor_core<T> &t) {
    *this = *this + t.derived();
    return *this;
  }
  // -=
  template <class T> tensor &operator-=(const tensor_core<T> &t) {
    *this = *this - t.derived();
    return *this;
  }

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

// vec_
template <class T, size_t N>
using vec_ = tensor<T, tensor_shape<size_t, const_size<N>>>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;

// vecx_
template <class T> using vecx_ = tensor<T, tensor_shape<size_t, size_t>>;
using vecx = vecx_<double>;

// mat_
template <class T, size_t M, size_t N>
using mat_ = tensor<T, tensor_shape<size_t, const_size<M>, const_size<N>>>;
using mat2 = mat_<double, 2, 2>;
using mat3 = mat_<double, 3, 3>;

// matx_
template <class T>
using matx_ = tensor<T, tensor_shape<size_t, size_t, size_t>>;
using matx = matx_<double>;

// cube_
template <class T, size_t M, size_t N, size_t L>
using cube_ =
    tensor<T,
           tensor_shape<size_t, const_size<M>, const_size<N>, const_size<L>>>;
using cube2 = cube_<double, 2, 2, 2>;
using cube3 = cube_<double, 3, 3, 3>;

// cubex_
template <class T>
using cubex_ = tensor<T, tensor_shape<size_t, size_t, size_t, size_t>>;
using cubex = matx_<double>;

// tstring
using tstring = vecx_<char>;
// wtstring
using wtstring = vecx_<wchar_t>;
// u16tstring
using u16tstring = vecx_<char16_t>;
// u32tstring
using u32tstring = vecx_<char32_t>;

// tensor_of_rank
namespace details {
template <class T, class SeqT> struct _make_tensor_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_tensor_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor<T, tensor_shape<size_t, always_t<size_t, Is>...>>;
};
}
template <class T, size_t Rank>
using tensor_of_rank = typename details::_make_tensor_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;
}