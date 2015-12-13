#pragma once

#include "base.hpp"

namespace wheels {

// remap_result
template <class ShapeT, class ET, class T, class MapFunT>
class remap_result
    : public tensor_op_result<ShapeT, ET, void,
                              remap_result<ShapeT, ET, T, MapT>> {
public:
  constexpr remap_result(T &&in, const ShapeT &s, MapFunT m, const ET &o)
      : _input(forward<T>(in)), _this_shape(s), _subs_this2input(m),
        _outlier_val(o) {}

  constexpr const ShapeT &shape() const { return _this_shape; }
  template <class... SubTs>
  constexpr decltype(auto) float_subs_in_input(const SubTs &... subs) const {
    return _subs_this2input(subs...);
  }

private:
  T _input;
  ShapeT _this_shape;
  MapFunT _subs_this2input;
  ET _outlier_val;
};



}
