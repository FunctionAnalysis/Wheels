#pragma once

#include <array>
#include <vector>

#include "tensor_frame.hpp"

namespace wheels {

    // array
    template <class ShapeT, class E, size_t N>
    class ts_element_reader<ts_category<ShapeT, std::array<E, N>>> 
        : public ts_element_readable<ts_category<ShapeT, std::array<E, N>>, true, false> {
    public:
        // read element at index
        template <class IndexT>
        constexpr const E & at_index_const_impl(const IndexT & index) const {
            return data_provider()[index];
        }
    };

    template <class ShapeT, class E, size_t N>
    class ts_element_writer<ts_category<ShapeT, std::array<E, N>>>
        : public ts_element_writable<ts_category<ShapeT, std::array<E, N>>, true, false> {
    public:
        // read element at index
        template <class IndexT>
        E & at_index_nonconst_impl(const IndexT & index) {
            return data_provider()[index];
        }
    };

    template <class ShapeT, class E, size_t N>
    class ts_category<ShapeT, std::array<E, N>>
        : public ts_category_inherit<ts_category<ShapeT, std::array<E, N>>> {
        static_assert(ShapeT::is_static, "");
        using base_t = ts_category_inherit<ts_category<ShapeT, std::array<E, N>>>;
    public:
        using shape_type = ShapeT;
        using data_provider_type = std::array<E, N>;
        using value_type = E;
    public:
        template <class ... EleTs>
        constexpr ts_category(EleTs && ... eles) 
            : base_t(ShapeT(), std::array<E, N>{ {(E)forward<EleTs>(eles) ...} }) {}
    };



}