#pragma once

#include <array>
#include <vector>
#include <amp.h>

#include "../core/types.hpp"
#include "../core/platforms.hpp"

#include "tensor_shape.hpp"

namespace wheels {

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
    template <class PlatformT, class T>
    struct is_element_accessible_at_index : no {};
    template <class PlatformT, class T>
    struct is_element_accessible_at_subs : no {};





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
        return { {(E)eles ...} };
    }
    
    template <class E, size_t N>
    struct is_constructible_with_shape_elements<std::array<E, N>> : yes {};
    
    template <class E, size_t N, class ShapeT, class ... EleTs>
    constexpr std::array<E, N> construct_with_shape_elements(types<std::array<E, N>>, const ShapeT & shape, const EleTs & ... eles) {
        return{ { (E)eles ... } };
    }

    // accessing elements
    template <class E, size_t N>
    struct is_element_accessible_at_index<platform_cpu, std::array<E, N>> : yes {};

    template <class E, size_t N, class IndexT>
    constexpr const E & element_at_index(const std::array<E, N> & a, const IndexT & index) {
        return a[index];
    }

    template <class E, size_t N, class IndexT>
    inline E & element_at_index(std::array<E, N> & a, const IndexT & index) {
        return a[index];
    }






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
        return { (E)eles... };
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
    struct is_element_accessible_at_index<platform_cpu, std::vector<E, AllocT>> : yes {};

    template <class E, class AllocT, class IndexT>
    inline const E & element_at_index(const std::vector<E, AllocT> & a, const IndexT & index) {
        return a[index];
    }
    template <class E, class AllocT, class IndexT>
    inline E & element_at_index(std::vector<E, AllocT> & a, const IndexT & index) {
        return a[index];
    }





    // concurrency::array_view
    // constructors
    template <class E, int Rank>
    struct is_constructible_with_shape<concurrency::array_view<E, Rank>> : yes {};

    namespace details {
        template <class E, int Rank, class ShapeT, int ... Is>
        inline concurrency::array_view<E, Rank> _construct_with_shape_seq(
            types<concurrency::array_view<E, 1>> t, 
            const ShapeT & shape,
            const_ints<int, Is...> seq) {
            return concurrency::array_view<E, Rank>(shape[const_int<Is>()] ...);
        }
    }
    
    template <class E, int Rank, class T, class ... SizeTs>
    inline concurrency::array_view<E, Rank> construct_with_shape(types<concurrency::array_view<E, Rank>> t, 
        const tensor_shape<T, SizeTs ...> & shape) {
        static_assert(Rank == sizeof...(SizeTs), "shape degree mismatch");
        return details::_construct_with_shape_seq(t, shape, make_const_sequence(const_int<Rank>()));
    }

    // accessing elements
    template <class E, int Rank>
    struct is_element_accessible_at_subs<platform_cpu, concurrency::array_view<E, Rank>> : yes {};
    
    template <class E, int Rank>
    struct is_element_accessible_at_subs<platform_amp, concurrency::array_view<E, Rank>> : yes {};
    
    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(const concurrency::array_view<E, Rank> & a, const SubTs & ... subs) restrict(cpu, amp) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }

    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(concurrency::array_view<E, Rank> & a, const SubTs & ... subs) restrict(cpu, amp) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }
    
}
