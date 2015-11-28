#pragma once

#include <array>
#include <vector>

#include "tensor_frame.hpp"

namespace wheels {

    // array
    namespace ts_props {

        // readable
        template <class ShapeT, class E, size_t N>
        struct readable_at_index<ts_category<ShapeT, std::array<E, N>>> : yes {};

        template <class E, size_t N, class IndexT>
        constexpr const E & at_index_const_impl(const std::array<E, N> & a, const IndexT & ind) {
            return a[ind];
        }

        // writable
        template <class ShapeT, class E, size_t N>
        struct writable_at_index<ts_category<ShapeT, std::array<E, N>>> : yes {};

        template <class E, size_t N, class IndexT>
        E & at_index_nonconst_impl(std::array<E, N> & a, const IndexT & ind) {
            return a[ind];
        }

        // const iteratable
        template <class ShapeT, class E, size_t N>
        struct const_iterator_type<ts_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::const_iterator;
        };

        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cbegin_impl(const ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cend_impl(const ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().end();
        }

        // nonconst iteratable
        template <class ShapeT, class E, size_t N>
        struct nonconst_iterator_type<ts_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::iterator;
        };

        template <class ShapeT, class E, size_t N>
        decltype(auto) begin_impl(ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        decltype(auto) end_impl(ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().end();
        }
    }
    
    template <class ShapeT, class E, size_t N>
    class ts_category<ShapeT, std::array<E, N>> 
        : public ts_category_base<ts_category<ShapeT, std::array<E, N>>> {
        using base_t = ts_category_base<ts_category<ShapeT, std::array<E, N>>>;
        static_assert(ShapeT::is_static, "");
    public:
        template <class ... EleTs>
        constexpr explicit ts_category(EleTs && ... eles)
            : base_t(ShapeT(), std::array<E, N>{{(E)forward<EleTs>(eles) ...}}) {}

        template <class CategoryT, bool RInd, bool RSub>
        constexpr ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) {
            assign_from(t);
        }

        constexpr ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };


    template <class E, size_t N>
    using vec_ = ts_category<tensor_shape<size_t, const_size<N>>, std::array<E, N>>;
    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;
    using vec4 = vec_<double, 4>;

    template <class E, size_t M, size_t N>
    using mat_ = ts_category<tensor_shape<size_t, const_size<M>, const_size<N>>, std::array<E, M * N>>;
    using mat2x2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3x2 = mat_<double, 3, 2>;
    using mat3x3 = mat_<double, 3, 3>;


    // vector

    namespace ts_props {

        // readable
        template <class ShapeT, class E, class AllocT>
        struct readable_at_index<ts_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class E, class AllocT, class IndexT>
        constexpr decltype(auto) at_index_const_impl(const std::vector<E, AllocT> & a, const IndexT & ind) {
            return a[ind];
        }

        // writable
        template <class ShapeT, class E, class AllocT>
        struct writable_at_index<ts_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class E, class AllocT, class IndexT>
        decltype(auto) at_index_nonconst_impl(std::vector<E, AllocT> & a, const IndexT & ind) {
            return a[ind];
        }

        // const iteratable
        template <class ShapeT, class E, class AllocT>
        struct const_iterator_type<ts_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::const_iterator;
        };

        template <class ShapeT, class E, class AllocT>
        constexpr decltype(auto) cbegin_impl(const ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        constexpr decltype(auto) cend_impl(const ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }

        // nonconst iteratable
        template <class ShapeT, class E, class AllocT>
        struct nonconst_iterator_type<ts_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::iterator;
        };

        template <class ShapeT, class E, class AllocT>
        decltype(auto) begin_impl(ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        decltype(auto) end_impl(ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }

        // reserve
        template <class ShapeT, class E, class AllocT>
        void reserve_impl(ts_base<ts_category<ShapeT, std::vector<E, AllocT>>> & t, size_t s) {
            t.data_provider().resize(s);
        }

    }



    template <class ShapeT, class E, class AllocT>
    class ts_category<ShapeT, std::vector<E, AllocT>>
        : public ts_category_base<ts_category<ShapeT, std::vector<E, AllocT>>> {
        using base_t = ts_category_base<ts_category<ShapeT, std::vector<E, AllocT>>>;
    public:
        template <class ... EleTs>
        constexpr explicit ts_category(const ShapeT & shape, EleTs && ... eles)
            : base_t(shape, std::vector<E, AllocT>({(E)forward<EleTs>(eles) ...})) {}

        template <class CategoryT, bool RInd, bool RSub>
        constexpr ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) {
            assign_from(t);
        }

        constexpr ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };




}