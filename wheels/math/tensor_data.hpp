#pragma once

#include <array>
#include <vector>
#include <amp.h>

#include "../core/types.hpp"
#include "../core/platforms.hpp"

#include "tensor_shape.hpp"

namespace wheels {

    // constructors
    // constexpr T construct
    using std::is_default_constructible;
    
    template <class T, class ... ArgTs>
    constexpr T construct_with_args(types<T>, ArgTs && ... args) {
        return T(forward<ArgTs>(args) ...);
    }
    template <class T, class ... ArgTs>
    constexpr T construct_with_args(types<T>, ArgTs && ... args) restrict(amp) {
        return T(forward<ArgTs>(args) ...);
    }



    // template <class T, class ShapeEleT, class ... ShapeSizeTs>
    // constexpr T construct_with_shape(types<T>, const tensor_shape<ShapeEleT, ShapeSizeTs ...> & shape);
    namespace details {
        template <class T, class ShapeT>
        struct _is_constructible_with_shape {
            template <class TT, class ShapeTT>
            static auto test(int) -> decltype(construct_with_shape(types<TT>(), std::declval<ShapeTT>()), yes()) {
                return yes();
            }
            template <class, class>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, ShapeT>(0))::value;
        };
    }
    template <class T, class ShapeT>
    struct is_constructible_with_shape 
        : const_bool<details::_is_constructible_with_shape<T, ShapeT>::value> {};


    // template <class T, class ... EleTs>
    // constexpr T construct_with_elements(types<T>, EleTs && ... eles);
    namespace details {
        template <class T, class ... EleTs>
        struct _is_constructible_with_elements {
            template <class TT, class ... EleTTs>
            static auto test(int) -> decltype(construct_with_elements(types<TT>(), std::declval<EleTTs>() ...), yes()) {
                return yes();
            }
            template <class, class ...>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, EleTs ...>(0))::value;
        };
    }
    template <class T, class ... EleTs>
    struct is_constructible_with_elements
        : const_bool<details::_is_constructible_with_elements<T, EleTs ...>::value> {};


    // template <class T, class ShapeT, class ... EleTs>
    // constexpr T construct_with_shape_elements(types<T>, const ShapeT & shape, EleTs && ... eles);
    namespace details {
        template <class T, class ShapeT, class ... EleTs>
        struct _is_constructible_with_shape_elements {
            template <class TT, class ShapeTT, class ... EleTTs>
            static auto test(int) -> decltype(construct_with_shape_elements(types<TT>(), std::declval<ShapeTT>(), std::declval<EleTs>() ...), yes()) {
                return yes();
            }
            template <class, class ...>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, ShapeT, EleTs ...>(0))::value;
        };
    }
    template <class T, class ShapeT, class ... EleTs>
    struct is_constructible_with_shape_elements
        : const_bool<details::_is_constructible_with_shape_elements<T, ShapeT, EleTs ...>::value> {};







    // element accessors
    
    // template <class T, class IndexT>
    // constexpr decltype(auto) element_at_index(T && data_provider, const IndexT & index);
    namespace details {
        template <class T, class IndexT>
        struct _is_element_accessible_at_index {
            template <class TT, class IndexTT>
            static auto test(int) -> decltype(element_at_index(std::declval<TT>(), std::declval<IndexTT>()), yes()) {
                return yes();
            }
            template <class, class>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, IndexT>(0))::value;
        };
        template <class T, class IndexT>
        struct _is_element_accessible_at_index_amp {
            template <class TT, class IndexTT>
            static auto test(int) restrict(cpu, amp) -> decltype(call_amp(element_at_index(std::declval<TT>(), std::declval<IndexTT>())), yes()){
                return yes();
            }
            template <class, class>
            static no test(...) restrict(cpu, amp) { return no(); }
            static constexpr bool value = decltype(test<T, IndexT>(0))::value;
        };
    }

    template <class PlatformT, class T, class IndexT>
    struct is_element_accessible_at_index : no {};

    template <class T, class IndexT>
    struct is_element_accessible_at_index<platform_cpu, T, IndexT>
        : const_bool<details::_is_element_accessible_at_index<T, IndexT>::value> {};
    template <class T, class IndexT>
    struct is_element_accessible_at_index<platform_amp, T, IndexT>
        : const_bool<details::_is_element_accessible_at_index_amp<T, IndexT>::value> {};




    // template <class T, class ... SubTs>
    // constexpr decltype(auto) element_at_subs(T && data_provider, const SubTs & ... index);
    namespace details {
        template <class T, class ... SubTs>
        struct _is_element_accessible_at_subs {
            template <class TT, class ... SubTTs>
            static auto test(int) -> decltype(element_at_subs(std::declval<TT>(), std::declval<SubTTs>() ...), yes()) {
                return yes();
            }
            template <class, class ...>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, SubTs ...>(0))::value;
        };
        template <class T, class ... SubTs>
        struct _is_element_accessible_at_subs_amp {
            template <class TT, class ... SubTTs>
            static auto test(int) -> decltype(call_amp(element_at_subs(std::declval<TT>(), std::declval<SubTTs>() ...)), yes()) {
                return yes();
            }
            template <class, class ...>
            static no test(...) { return no(); }
            static constexpr bool value = decltype(test<T, SubTs ...>(0))::value;
        };
    }

    template <class PlatformT, class T, class ... SubTs>
    struct is_element_accessible_at_subs : no {};

    template <class T, class ... SubTs>
    struct is_element_accessible_at_subs<platform_cpu, T, SubTs ...>
        : const_bool<details::_is_element_accessible_at_subs<T, SubTs ...>::value> {};
    template <class T, class ... SubTs>
    struct is_element_accessible_at_subs<platform_amp, T, SubTs ...>
        : const_bool<details::_is_element_accessible_at_subs_amp<T, SubTs ...>::value> {};










    // predefinitions

    // std::array
    // constructors
    template <class E, size_t N, class ShapeT>
    constexpr std::array<E, N> construct_with_shape(types<std::array<E, N>>, const ShapeT & shape) {
        return std::array<E, N>();
    }
    template <class E, size_t N, class ... EleTs>
    constexpr std::array<E, N> construct_with_elements(types<std::array<E, N>>, const EleTs & ... eles) {
        return { {(E)eles ...} };
    }
    template <class E, size_t N, class ShapeT, class ... EleTs>
    constexpr std::array<E, N> construct_with_shape_elements(types<std::array<E, N>>, const ShapeT & shape, const EleTs & ... eles) {
        return{ { (E)eles ... } };
    }

    // element accessors
    template <class E, size_t N, class IndexT>
    constexpr const E & element_at_index(const std::array<E, N> & a, const IndexT & index) {
        return a[index];
    }
    template <class E, size_t N, class IndexT>
    inline E & element_at_index(std::array<E, N> & a, const IndexT & index) {
        return a[index];
    }



    // concurrency::array_view
    // constructors
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

    // element accessors
    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(const concurrency::array_view<E, Rank> & a, const SubTs & ... subs) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }
    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(concurrency::array_view<E, Rank> & a, const SubTs & ... subs) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }
    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(const concurrency::array_view<E, Rank> & a, const SubTs & ... subs) restrict(amp) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }
    template <class E, int Rank, class ... SubTs>
    inline decltype(auto) element_at_subs(concurrency::array_view<E, Rank> & a, const SubTs & ... subs) restrict(amp) {
        static_assert(Rank == sizeof...(subs), "subscript count mismatch");
        const int subs_arr[] = { subs ... };
        concurrency::index<Rank> ind(subs_arr);
        return a[ind];
    }
    
}
