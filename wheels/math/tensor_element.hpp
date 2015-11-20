#pragma once

#include <array>
#include <vector>

#include "tensor_shape.hpp"

namespace wheels {

    namespace tensor_methods {

        // constructors
        template <class T, class ... ArgTs>
        constexpr T construct_with_args(ArgTs && ... args) {
            return T(forward<ArgTs>(args) ...);
        }
        template <class T, class ShapeEleT, class ... ShapeSizeTs>
        constexpr T construct_with_shape(const tensor_shape<ShapeEleT, ShapeSizeTs ...> & shape) {
            static_assert(always<bool, false, T>::value, "not implemented");
        }        
        template <class T, class ... EleTs>
        constexpr T construct_with_elements(EleTs && ... eles) {
            static_assert(always<bool, false, T>::value, "not implemented");
        }
        template <class T, class ShapeT, class ... EleTs>
        constexpr T construct_with_shape_elements(const ShapeT & shape, EleTs && ... eles) {
            static_assert(always<bool, false, T>::value, "not implemented");
        }


        // element accessors
        template <class T, class IndexT>
        constexpr decltype(auto) element_at_index(T && storage, const IndexT & index) {
            return forward<T>(storage)[index];
        }

    }

}
