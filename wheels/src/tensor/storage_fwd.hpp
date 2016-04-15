#pragma once

namespace wheels {
constexpr struct _with_elements {
} with_elements;
constexpr struct _with_iterators {
} with_iterators;

template <class T, class ShapeT, bool ShapeIsStatic = ShapeT::is_static>
class storage;
}
