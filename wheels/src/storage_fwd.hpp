#pragma once

namespace wheels {
struct _with_elements {
  constexpr _with_elements() {}
};
constexpr _with_elements with_elements = {};
struct _with_iterators {
  constexpr _with_iterators() {}
};
constexpr _with_iterators with_iterators = {};

template <class T, class ShapeT, bool ShapeIsStatic = ShapeT::is_static>
class storage;

template <class T, class ShapeT, bool ShapeIsStatic = ShapeT::is_static>
class map_storage;
}
