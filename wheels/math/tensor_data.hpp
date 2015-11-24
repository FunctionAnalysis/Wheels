#pragma once

#include <array>
#include <vector>
#include <amp.h>

#include "../core/types.hpp"
#include "../core/platforms.hpp"

#include "tensor_shape.hpp"

namespace wheels {

    // tensor data provider
    namespace tdp {

        // constructors
        using std::is_default_constructible;
        using std::is_copy_constructible;
        using std::is_move_constructible;

        template <class T, class ... ArgTs>
        constexpr T construct_with_args(types<T>, ArgTs && ... args) {
            return T(forward<ArgTs>(args) ...);
        }
        template <class T, class ... ArgTs>
        constexpr T construct_with_args(types<T>, ArgTs && ... args) restrict(amp) {
            return T(forward<ArgTs>(args) ...);
        }

        template <class T>
        struct is_constructible_with_shape : no {};
        template <class T>
        struct is_constructible_with_elements : no {};
        template <class T>
        struct is_constructible_with_shape_elements : no {};




        // accessing elements
        template <class T>
        struct is_element_readable_at_index : no {};
        template <class T>
        struct is_element_writable_at_index : no {};

        template <class T>
        struct is_element_readable_at_subs : no {};
        template <class T>
        struct is_element_writable_at_subs : no {};



        // predefinitions

        // std::array
        // constructors
        template <class E, size_t N>
        struct is_constructible_with_shape<std::array<E, N>> : yes {};

        template <class E, size_t N, class ShapeT>
        constexpr std::array<E, N> construct_with_shape(types<std::array<E, N>>, const ShapeT & shape) {
            return std::array<E, N>();
        }

        template <class E, size_t N>
        struct is_constructible_with_elements<std::array<E, N>> : yes {};

        template <class E, size_t N, class ... EleTs>
        constexpr std::array<E, N> construct_with_elements(types<std::array<E, N>>, const EleTs & ... eles) {
            return{ {(E)eles ...} };
        }

        template <class E, size_t N>
        struct is_constructible_with_shape_elements<std::array<E, N>> : yes {};

        template <class E, size_t N, class ShapeT, class ... EleTs>
        constexpr std::array<E, N> construct_with_shape_elements(types<std::array<E, N>>, const ShapeT & shape, const EleTs & ... eles) {
            return{ { (E)eles ... } };
        }

        // accessing elements
        template <class E, size_t N>
        struct is_element_readable_at_index<std::array<E, N>> : yes {};

        template <class E, size_t N, class IndexT>
        constexpr const E & element_at_index(const std::array<E, N> & a, const IndexT & index) {
            return a[index];
        }

        template <class E, size_t N>
        struct is_element_writable_at_index<std::array<E, N>> : yes {};

        template <class E, size_t N, class IndexT>
        inline E & element_at_index(std::array<E, N> & a, const IndexT & index) {
            return a[index];
        }

        // reserve
        template <class E, size_t N, class ShapeT>
        inline void reserve_storage(const ShapeT & shape, std::array<E, N> &) {}





        // std::vector
        // constructors
        template <class E, class AllocT>
        struct is_constructible_with_shape<std::vector<E, AllocT>> : yes {};

        template <class E, class AllocT, class ShapeT>
        inline std::vector<E, AllocT> construct_with_shape(types<std::vector<E, AllocT>>, const ShapeT & shape) {
            return std::vector<E, AllocT>(shape.magnitude());
        }

        template <class E, class AllocT>
        struct is_constructible_with_elements<std::vector<E, AllocT>> : yes {};

        template <class E, class AllocT, class ... EleTs>
        inline std::vector<E, AllocT> construct_with_elements(types<std::vector<E, AllocT>>, const EleTs & ... eles) {
            return{ (E)eles... };
        }

        template <class E, class AllocT>
        struct is_constructible_with_shape_elements<std::vector<E, AllocT>> : yes {};

        template <class E, class AllocT, class ShapeT, class ... EleTs>
        inline std::vector<E, AllocT> construct_with_shape_elements(types<std::vector<E, AllocT>>, const ShapeT & shape, const EleTs & ... eles) {
            std::vector<E, AllocT> v = { (E)eles ... };
            v.resize(shape.magnitude());
            return v;
        }

        // accessing elements
        template <class E, class AllocT>
        struct is_element_readable_at_index<std::vector<E, AllocT>> : yes {};

        template <class E, class AllocT, class IndexT>
        inline decltype(auto) element_at_index(const std::vector<E, AllocT> & a, const IndexT & index) {
            return a[index];
        }

        template <class E, class AllocT>
        struct is_element_writable_at_index<std::vector<E, AllocT>> : yes {};

        template <class E, class AllocT, class IndexT>
        inline decltype(auto) element_at_index(std::vector<E, AllocT> & a, const IndexT & index) {
            return a[index];
        }

        // reserve
        template <class E, class AllocT, class ShapeT>
        inline void reserve_storage(const ShapeT & shape, std::vector<E, AllocT> & a) {
            a.resize(shape.magnitude());
        }

    }
    
}
