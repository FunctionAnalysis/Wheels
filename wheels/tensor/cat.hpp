#pragma once

#include "tensor.hpp"

namespace wheels {

// cat_result
template <class ShapeT, class ET, size_t Axis, class T, class... Ts>
class cat_result
    : public tensor_op_result_base<ShapeT, ET, void,
                                   cat_result<ShapeT, ET, Axis, T, Ts...>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;

  cat_result(T &&in, Ts &&... ins)
      : _inputs(forward<T>(in), forward<Ts>(ins)...) {}
  constexpr decltype(auto) inputs() const {return _inputs;}
  constexpr auto shape() const {return make_shape() }

private:
  std::tuple<T, Ts...> _inputs;
};

// shape_of
template <class ShapeT, class ET, size_t Axis, class T, class... Ts>
constexpr auto shape_of(const cat_result<ShapeT, ET, Axis, T, Ts...> &m) {
  throw std::runtime_error("not implemented yet");
}

}
