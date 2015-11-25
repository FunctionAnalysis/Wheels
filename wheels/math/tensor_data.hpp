#pragma once

#include <array>
#include <vector>
#include <amp.h>

#include "../core/types.hpp"

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


        // iteration
        template <class T>
        struct is_selective_iteratable : no {};




        // predefinitions

        // std::array
        // constructors
        template <class E, size_t N>
        struct is_constructible_with_shape<std::array<E, N>> : yes {};

        
        template <class E, size_t N, size_t ... Is>
        constexpr std::array<E, N> _construct_std_array_seq(types<std::array<E, N>>, const_ints<size_t, Is...>) {
            return{ { always<int, 0, const_index<Is>>::value ... } };
        }
        template <class E, size_t N, class ShapeT>
        constexpr std::array<E, N> construct_with_shape(types<std::array<E, N>> t, const ShapeT & shape) {
            return _construct_std_array_seq(t, make_const_sequence(const_size<N>()));
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





        // dictionary
        template <class IndexT, class T, class InternalT = std::map<IndexT, T>>
        struct dictionary {
            using value_type = T;
            InternalT internal;
            template <class Archiver>
            void serialize(Archiver & ar) {
                ar(internal);
            }
        };

        template <class IndexT, class T, class InternalT>
        struct is_constructible_with_shape<dictionary<IndexT, T, InternalT>> : yes {};

        template <class IndexT, class T, class InternalT, class ShapeT>
        inline dictionary<IndexT, T, InternalT> construct_with_shape(types<dictionary<IndexT, T, InternalT>>, 
            const ShapeT & shape) {
            return dictionary<IndexT, T, InternalT>();
        }

        template <class IndexT, class T, class InternalT>
        struct is_constructible_with_elements<dictionary<IndexT, T, InternalT>> : yes {};

        template <class IndexT, class T, class InternalT, class ... EleTs>
        inline dictionary<IndexT, T, InternalT> construct_with_elements(types<dictionary<IndexT, T, InternalT>>, 
            const EleTs & ... eles) {
            dictionary<IndexT, T, InternalT> d;
            const std::array<T, sizeof...(EleTs)> eles_array = { { (T)eles... } };
            for (IndexT i = 0; i < eles_array.size(); i++) {
                if (eles_array[i]) {
                    d.internal.emplace(i, eles_array[i]);
                }
            }
            return d;
        }

        template <class IndexT, class T, class InternalT>
        struct is_constructible_with_shape_elements<dictionary<IndexT, T, InternalT>> : yes {};

        template <class IndexT, class T, class InternalT, class ShapeT, class ... EleTs>
        inline dictionary<IndexT, T, InternalT> construct_with_shape_elements(types<dictionary<IndexT, T, InternalT>> t,
            const ShapeT & shape, const EleTs & ... eles) {
            return construct_with_elements(t, eles ...);
        }

        // accessing elements
        template <class IndexT, class T, class InternalT>
        struct is_element_readable_at_index<dictionary<IndexT, T, InternalT>> : yes {};

        template <class IndexT, class T, class InternalT>
        inline T element_at_index(const dictionary<IndexT, T, InternalT> & a, const IndexT & index) {
            return a.internal.find(index) == a.internal.end() ? T() : a.internal.at(index);
        }

        template <class IndexT, class T, class InternalT>
        struct is_element_writable_at_index<dictionary<IndexT, T, InternalT>> : yes {};

        template <class IndexT, class T, class InternalT>
        inline T & element_at_index(dictionary<IndexT, T, InternalT> & a, const IndexT & index) {
            return a.internal[index];
        }


    }
    
}
