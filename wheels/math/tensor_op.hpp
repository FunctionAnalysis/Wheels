#pragma once

#include "tensor_data.hpp"
#include "tensor.hpp"

namespace wheels {

    namespace tdp {

        // constant
        template <class T>
        struct constant {
            using value_type = T;
            T val;
            template <class TT>
            constexpr constant(TT && v) : val(forward<TT>(v)) {}
        };

        // accessing elements
        template <class T>
        struct is_element_readable_at_index<constant<T>> : yes {};

        template <class T, class IndexT>
        constexpr const T & element_at_index(const constant<T> & a, const IndexT & index) {
            return a.val;
        }

        template <class T>
        struct is_element_writable_at_index<constant<T>> : yes {};

        template <class T, class IndexT>
        T & element_at_index(constant<T> & a, const IndexT & index) {
            return a.val;
        }

        template <class T>
        struct is_element_readable_at_subs<constant<T>> : yes {};

        template <class T, class ... SubTs>
        constexpr const T & element_at_subs(const constant<T> & a, const SubTs & ...) {
            return a.val;
        }

        template <class T>
        struct is_element_writable_at_subs<constant<T>> : yes {};

        template <class T, class ... SubTs>
        T & element_at_subs(constant<T> & a, const SubTs & ...) {
            return a.val;
        }

        // reserve
        template <class T, class ShapeT>
        inline void reserve_storage(const ShapeT & shape, constant<T> &) {}
    }

    template <class T, class ST, class ... SizeTs>
    constexpr auto constants(const tensor_shape<ST, SizeTs...> & shape, T && val) {
        return compose_tensor_layout(shape, tdp::constant<std::decay_t<T>>(forward<T>(val)));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor_layout(shape, tdp::constant<T>(0));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor_layout(shape, tdp::constant<T>(1));
    }




}
