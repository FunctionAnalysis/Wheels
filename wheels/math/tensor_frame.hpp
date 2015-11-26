#pragma once

#include "tensor_shape.hpp"

namespace wheels {

    template <class ShapeT, class DataProviderT, bool StaticShape = ShapeT::is_static> class ts_layout;

    class ts_core {};

    template <class LayoutT>
    class ts_base : public ts_core {
    public:
        using layout_type = LayoutT;

        // layout
        constexpr const layout_type & layout() const { return (const layout_type &)(*this); }
        layout_type & layout() { return (layout_type &)(*this); }

        // shape()
        constexpr decltype(auto) shape() const { return layout().shape_impl(); }
        // rank()
        constexpr auto rank() const { return const_ints<int, decltype(shape())::rank>(); }
        // size(...)
        template <class T, T Idx>
        constexpr auto size(const const_ints<T, Idx> & i) const {
            static_assert(Idx >= 0 && Idx < rank(), "invalid Idx");
            return shape().at(i);
        }
        // numel()
        auto numel() const { return shape().magnitude(); }

        // data_provider()
        constexpr decltype(auto) data_provider() const { return layout().data_provider_impl(); }
        decltype(auto) data_provider() { return layout().data_provider_impl(); }

    };

    // ele reader base/impl
    template <class LayoutT> using ts_ele_reader_inherit = ts_base<LayoutT>;
    template <class LayoutT> class ts_ele_reader_impl;
    template <class LayoutT, bool EleReadableAtIndex, bool EleReadableAtSubs>
    class ts_ele_reader_base : public ts_ele_reader_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_ele_reader_base<LayoutT, true, true> : public ts_ele_reader_inherit<LayoutT> {
        const ts_ele_reader_impl<LayoutT> & impl() const { 
            return static_cast<const ts_ele_reader_impl<LayoutT> &>(*this); 
        }
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return impl().at_index_const_impl(index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_subs_const_impl(subs ...);
        }
    };
    template <class LayoutT>
    class ts_ele_reader_base<LayoutT, true, false> : public ts_ele_reader_inherit<LayoutT> {
        const ts_ele_reader_impl<LayoutT> & impl() const {
            return static_cast<const ts_ele_reader_impl<LayoutT> &>(*this);
        }
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return impl().at_index_const_impl(index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_index_const_impl(sub2ind(shape(), subs ...));
        }
    };
    template <class LayoutT>
    class ts_ele_reader_base<LayoutT, false, true> : public ts_ele_reader_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_ele_reader_impl : public ts_ele_reader_base<LayoutT, false, false> {};

    // ele writer base/impl
    template <class LayoutT> using ts_ele_writer_inherit = ts_ele_reader_impl<LayoutT>;
    template <class LayoutT, bool EleWritableAtIndex, bool EleWritableAtSubs>
    class ts_ele_writer_base : public ts_ele_writer_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_ele_writer_base<LayoutT, true, true> : public ts_ele_writer_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_ele_writer_base<LayoutT, true, false> : public ts_ele_writer_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_ele_writer_base<LayoutT, false, true> : public ts_ele_writer_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_ele_writer_impl : public ts_ele_writer_base<LayoutT, false, false> {};

    // const iterator base/impl
    template <class LayoutT> using ts_const_iterater_inherit = ts_ele_writer_impl<LayoutT>;
    template <class LayoutT, bool HasConstIterator>
    class ts_const_iterater_base : public ts_const_iterater_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_const_iterater_base<LayoutT, true> : public ts_const_iterater_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_const_iterater_impl : public ts_const_iterater_base<LayoutT, false> {};

    // const iterator base/impl
    template <class LayoutT> using ts_nonconst_iterater_inherit = ts_const_iterater_impl<LayoutT>;
    template <class LayoutT, bool HasNonConstIterator>
    class ts_nonconst_iterater_base : public ts_nonconst_iterater_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_nonconst_iterater_base<LayoutT, true> : public ts_nonconst_iterater_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_nonconst_iterater_impl : public ts_nonconst_iterater_base<LayoutT, false> {};

    // const non zero const iterator base/impl
    template <class LayoutT> using ts_non_zero_const_iterator_inherit = ts_nonconst_iterater_impl<LayoutT>;
    template <class LayoutT, bool HasNonZeroConstIterator>
    class ts_non_zero_const_iterator_base : public ts_non_zero_const_iterator_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_non_zero_const_iterator_base<LayoutT, true> : public ts_non_zero_const_iterator_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_non_zero_const_iterator_impl : public ts_non_zero_const_iterator_base<LayoutT, false> {};

    // reduce base/impl
    template <class LayoutT> using ts_reduce_inherit = ts_non_zero_const_iterator_impl<LayoutT>;
    template <class LayoutT, bool Reducable>
    class ts_reduce_base : public ts_reduce_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_reduce_base<LayoutT, true> : public ts_reduce_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_reduce_impl : public ts_reduce_base<LayoutT, false> {};

    // reserve base/impl
    template <class LayoutT> using ts_reserve_inherit = ts_reduce_impl<LayoutT>;
    template <class LayoutT, bool Reservable>
    class ts_reserve_base : public ts_reserve_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_reserve_base<LayoutT, true> : public ts_reserve_inherit<LayoutT> {

    };
    template <class LayoutT>
    class ts_reserve_impl : public ts_reserve_base<LayoutT, false> {};

    // assign base/impl
    template <class LayoutT> using ts_assign_inherit = ts_reserve_impl<LayoutT>;
    template <class LayoutT>
    class ts_assign_base : public ts_assign_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_assign_impl : public ts_assign_base<LayoutT> {};

    // ext base/impl
    template <class LayoutT> using ts_ext_inherit = ts_assign_impl<LayoutT>;
    template <class LayoutT>
    class ts_ext_base : public ts_ext_inherit<LayoutT> {};
    template <class LayoutT>
    class ts_ext_impl : public ts_ext_base<LayoutT> {};


    // layout
    template <class LayoutT> using ts_layout_inherit = ts_ext_impl<LayoutT>;
    
    template <class ShapeT, class DataProviderT>
    class ts_layout<ShapeT, DataProviderT, true> 
        : public ts_layout_inherit<ts_layout<ShapeT, DataProviderT, true>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
    public:
        template <class DPT>
        constexpr ts_layout(const ShapeT & s, DPT && dp) : _data_provider(forward<DPT>(dp)) {}


    public:
        constexpr ShapeT shape_impl() const { return ShapeT(); }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }
    private:
        DataProviderT _data_provider;
    };

    template <class ShapeT, class DataProviderT>
    class ts_layout<ShapeT, DataProviderT, false>
        : public ts_layout_inherit<ts_layout<ShapeT, DataProviderT, false>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
    public:
        template <class DPT>
        constexpr ts_layout(const ShapeT & s, DPT && dp) : _shape(s), _data_provider(forward<DPT>(dp)) {}

    public:
        constexpr const ShapeT & shape_impl() const { return _shape; }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }
    private:
        ShapeT _shape;
        DataProviderT _data_provider;
    };


}