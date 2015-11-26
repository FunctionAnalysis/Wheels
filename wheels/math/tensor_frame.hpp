#pragma once

#include "tensor_shape.hpp"

namespace wheels {

    template <class ShapeT, class DataProviderT> 
    class ts_category;

    template <class CategoryT>
    class ts_base {
    public:
        using layout_type = CategoryT;

        // layout
        constexpr const layout_type & layout() const { return static_cast<const layout_type &>(*this); }
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

#define WHEELS_DEFINE_TS_FRAME_IMPL(name) \
    constexpr const name##_impl<CategoryT> & impl() const { \
        return static_cast<const name##_impl<CategoryT> &>(*this); \
    } \
    name##_impl<CategoryT> & impl() { \
        return static_cast<name##_impl<CategoryT> &>(*this); \
    } 



    // ele reader base/impl
    template <class CategoryT> 
    using ts_ele_reader_inherit = ts_base<CategoryT>;

    template <class CategoryT> class ts_ele_reader_impl;
    template <class CategoryT, bool EleReadableAtIndex, bool EleReadableAtSubs>
    class ts_ele_reader : public ts_ele_reader_inherit<CategoryT> {
    public:
        template <class IndexT>
        constexpr int at_index_const(const IndexT & index) const {
            static_assert(always<bool, false, IndexT>::value,
                "the data prodiver is not readable, "
                "explicit instantate ts_ele_reader_impl<CategoryT> and "
                "implement at_index_const_impl(...) and/or at_subs_const_impl(...) to fix this");
        }
        template <class ... SubTs>
        constexpr int at_subs_const(const SubTs & ... subs) const {
            static_assert(always<bool, false, SubTs ...>::value, 
                "the data prodiver is not readable, "
                "explicit instantate ts_ele_reader_impl<CategoryT> and "
                "implement at_index_const_impl(...) and/or at_subs_const_impl(...) to fix this");
        }
    };
    template <class CategoryT>
    class ts_ele_reader<CategoryT, true, true> : public ts_ele_reader_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_reader)
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
    template <class CategoryT>
    class ts_ele_reader<CategoryT, true, false> : public ts_ele_reader_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_reader)
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
    template <class CategoryT>
    class ts_ele_reader<CategoryT, false, true> : public ts_ele_reader_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_reader)
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & ... subs) {
                return impl().at_subs_const_impl(subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_subs_const_impl(subs ...);
        }
    };

    template <class CategoryT> 
    class ts_ele_reader_impl : public ts_ele_reader<CategoryT, false, false> {};





    // ele writer base/impl
    template <class CategoryT> 
    using ts_ele_writer_inherit = ts_ele_reader_impl<CategoryT>;

    template <class CategoryT> class ts_ele_writer_impl;
    template <class CategoryT, bool EleWritableAtIndex, bool EleWritableAtSubs>
    class ts_ele_writer : public ts_ele_writer_inherit<CategoryT> {
    public:
        template <class IndexT>
        constexpr decltype(auto) at_index_nonconst(const IndexT & index) const {
            return at_index_const(index);
        }
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_nonconst(const SubTs & ... subs) const {
            return at_subs_const(subs ...);
        }
    };
    template <class CategoryT>
    class ts_ele_writer<CategoryT, true, true> : public ts_ele_writer_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_writer)
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return impl().at_index_nonconst_impl(index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_subs_nonconst_impl(subs ...);
        }
    };
    template <class CategoryT>
    class ts_ele_writer<CategoryT, true, false> : public ts_ele_writer_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_writer)
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return impl().at_index_nonconst_impl(index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_index_nonconst_impl(sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class ts_ele_writer<CategoryT, false, true> : public ts_ele_writer_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_ele_writer)
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return impl().at_subs_nonconst_impl(subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return impl().at_subs_nonconst_impl(subs ...);
        }
    };
    
    template <class CategoryT>
    class ts_ele_writer_impl : public ts_ele_writer<CategoryT, false, false> {};





    // const iterator base/impl
    template <class CategoryT> 
    using ts_const_iterater_inherit = ts_ele_writer_impl<CategoryT>;

    template <class CategoryT> class ts_const_iterater_impl;
    template <class CategoryT, class ConstIterT>
    class ts_const_iterater : public ts_const_iterater_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_const_iterater)
    public:
        constexpr ConstIterT cbegin() const { return impl().cbegin_impl(); }
        constexpr ConstIterT begin() const { return cbegin(); }
        constexpr ConstIterT cend() const { return impl().cend_impl(); }
        constexpr ConstIterT end() const { return cend(); }
    };
    template <class CategoryT>
    class ts_const_iterater<CategoryT, void> : public ts_const_iterater_inherit<CategoryT> {};

    namespace details {
        template <class CategoryT>
        struct _ts_const_iter : std::iterator<std::random_access_iterator_tag,
            typename CategoryT::value_type,
            std::ptrdiff_t> {
            const ts_const_iterater_impl<CategoryT> & self;
            size_t ind;
            constexpr _ts_const_iter(const ts_const_iterater_impl<CategoryT> & s, size_t i = 0) : self(s), ind(i) {}
            constexpr decltype(auto) operator * () const { return self.at_index_const(ind); }
            constexpr decltype(auto) operator -> () const { return self.at_index_const(ind); }
            _ts_const_iter & operator ++() { ++ind; return *this; }
            _ts_const_iter & operator --() { assert(ind != 0);  --ind; return *this; }
            _ts_const_iter & operator +=(size_t s) { ind += s; return *this; }
            _ts_const_iter & operator -=(size_t s) { ind -= s; return *this; }
            constexpr _ts_const_iter operator + (size_t s) const { return _ts_const_iter(self, ind + s); }
            constexpr _ts_const_iter operator - (size_t s) const { return _ts_const_iter(self, ind - s); }
            std::ptrdiff_t operator - (const _ts_const_iter & it) const { return ind - it.ind; }
            constexpr bool operator == (const _ts_const_iter & it) const {
                assert(&self == &(it.self));
                return ind == it.ind;
            }
            constexpr bool operator != (const _ts_const_iter & it) const {
                return ind != it.ind;
            }
            constexpr bool operator < (const _ts_const_iter & it) const {
                return ind < it.ind;
            }
        };
    }
    template <class CategoryT>
    class ts_const_iterater_impl : public ts_const_iterater<CategoryT, details::_ts_const_iter<CategoryT>> {
    public:
        using const_iterator = details::_ts_const_iter<CategoryT>;
        constexpr const_iterator cbegin_impl() const { return const_iterator(*this, 0); }
        constexpr const_iterator cend_impl() const { return const_iterator(*this, numel()); }
    };





    // nonconst iterator base/impl
    template <class CategoryT> 
    using ts_nonconst_iterater_inherit = ts_const_iterater_impl<CategoryT>;

    template <class CategoryT> class ts_nonconst_iterater_impl;
    template <class CategoryT, class NonConstIterT>
    class ts_nonconst_iterater : public ts_nonconst_iterater_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_nonconst_iterater)
    public:
        NonConstIterT begin() { return impl().begin_impl(); }
        NonConstIterT end() { return impl().end_impl(); }
    };
    template <class CategoryT>
    class ts_nonconst_iterater<CategoryT, void> : public ts_nonconst_iterater_inherit<CategoryT> {
    public:
        decltype(auto) begin() { return cbegin(); }
        decltype(auto) end() { return cend(); }
    };

    namespace details {
        template <class CategoryT>
        struct _ts_nonconst_iter : std::iterator<std::random_access_iterator_tag,
            typename CategoryT::value_type,
            std::ptrdiff_t> {
            ts_const_iterater_impl<CategoryT>  & self;
            size_t ind;
            constexpr _ts_nonconst_iter(ts_const_iterater_impl<CategoryT> & s, size_t i = 0) : self(s), ind(i) {}
            decltype(auto) operator * () const { return self.at_index_nonconst(ind); }
            decltype(auto) operator -> () const { return self.at_index_nonconst(ind); }
            _ts_nonconst_iter & operator ++() { ++ind; return *this; }
            _ts_nonconst_iter & operator --() { assert(ind != 0);  --ind; return *this; }
            _ts_nonconst_iter & operator +=(size_t s) { ind += s; return *this; }
            _ts_nonconst_iter & operator -=(size_t s) { ind -= s; return *this; }
            constexpr _ts_nonconst_iter operator + (size_t s) const { return _ts_nonconst_iter(self, ind + s); }
            constexpr _ts_nonconst_iter operator - (size_t s) const { return _ts_nonconst_iter(self, ind - s); }
            std::ptrdiff_t operator - (const _ts_nonconst_iter & it) const { return ind - it.ind; }
            constexpr bool operator == (const _ts_nonconst_iter & it) const {
                assert(&self == &(it.self));
                return ind == it.ind;
            }
            constexpr bool operator != (const _ts_nonconst_iter & it) const {
                return ind != it.ind;
            }
            constexpr bool operator < (const _ts_nonconst_iter & it) const {
                return ind < it.ind;
            }
        };
    }
    template <class CategoryT>
    class ts_nonconst_iterater_impl : public ts_nonconst_iterater<CategoryT, details::_ts_nonconst_iter<CategoryT>> {
    public:
        using iterator = details::_ts_nonconst_iter<CategoryT>;
        constexpr iterator begin_impl() const { return iterator(*this, 0); }
        constexpr iterator end_impl() const { return iterator(*this, numel()); }
    };





    // const nonzero iterator base/impl
    template <class CategoryT> 
    using ts_nonzero_iterator_inherit = ts_nonconst_iterater_impl<CategoryT>;

    template <class CategoryT> class ts_nonzero_iterator_impl;
    template <class CategoryT, class NZIteratorT>
    class ts_nonzero_iterator : public ts_nonzero_iterator_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_nonzero_iterator)
    public:
        constexpr NZIteratorT nzbegin() const { return impl().nzbegin_impl(); }
        constexpr NZIteratorT nzend() const { return impl().nzend_impl(); }
    };
    template <class CategoryT>
    class ts_nonzero_iterator<CategoryT, void> : public ts_nonzero_iterator_inherit<CategoryT> {};

    template <class CategoryT>
    class ts_nonzero_iterator_impl : public ts_nonzero_iterator<CategoryT, void> {};





    // reduce base/impl
    template <class CategoryT> 
    using ts_reduce_inherit = ts_nonzero_iterator_impl<CategoryT>;

    template <class CategoryT> class ts_reduce_impl;
    template <class CategoryT, bool Reducable>
    class ts_reduce : public ts_reduce_inherit<CategoryT> {};
    template <class CategoryT>
    class ts_reduce<CategoryT, true> : public ts_reduce_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_reduce)
    public:
        template <class T, class ReduceT>
        constexpr T reduce(const T & initial, ReduceT && reducer) const {
            return impl().reduce_impl(initial, forward<ReduceT>(reducer));
        }
        auto sum() const { return reduce(typename CategoryT::value_type(0), binary_op_plus()); }
        auto mean() const { return sum() / numel(); }
        auto prod() const { return reduce(typename CategoryT::value_type(1), binary_op_mul()); }
        bool all() const { return reduce(true, binary_op_and()); }
        bool not_all() const { return !all(); }
        bool any() const { return reduce(false, binary_op_or()); }
        bool none() const { return !any(); }
    };

    template <class CategoryT>
    class ts_reduce_impl : public ts_reduce<CategoryT, false> {
        static constexpr size_t _parallel_thres = 70000;
        static constexpr size_t _parallel_batch = 35000;
    public:
        template <class T, class ReduceT>
        T reduce_impl(const T & base, ReduceT && reducer) const {
            T result = base;
            if (numel() < _parallel_thres) {
                for (size_t ind = 0; ind < numel(); ind++) {
                    result = reducer(result, at_index_const(ind));
                }
            } else {
                result = parallel_reduce(cbegin(), cend(), 
                    base, forward<ReduceT>(reducer),
                    _parallel_batch);
            }
            return result;
        }
    };





    // reserve base/impl
    template <class CategoryT> using ts_reserve_inherit = ts_reduce_impl<CategoryT>;
    
    template <class CategoryT> class ts_reserve_impl;
    template <class CategoryT>
    class ts_reserve : public ts_reserve_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_reserve)
    public:
        template <class SizeT>
        void reserve(const SizeT & s) {
            impl().reserve_impl(s);
        }
    };

    template <class CategoryT>
    class ts_reserve_impl : public ts_reserve<CategoryT> {
    public:
        template <class SizeT> void reserve_impl(const SizeT &) {}
    };




    // assign base/impl
    template <class CategoryT> using ts_assign_inherit = ts_reserve_impl<CategoryT>;
    template <class CategoryT> class ts_assign_impl;
    template <class CategoryT>
    class ts_assign : public ts_assign_inherit<CategoryT> {
        WHEELS_DEFINE_TS_FRAME_IMPL(ts_assign)
    public:
        template <class LayoutFromT>
        void assign_from(const ts_base<LayoutFromT> & from) { impl().assign_from_impl(to); }
    };
    template <class CategoryT>
    class ts_assign_impl : public ts_assign<CategoryT> {
        static constexpr size_t _parallel_thres = 70000;
        static constexpr size_t _parallel_batch = 35000;
    public:
        template <class LayoutFromT>
        void assign_from_impl(const ts_base<LayoutFromT> & from) {
            assert(shape() == to.shape());
            reserve(numel());
            if (numel() < _parallel_thres) {
                for (size_t ind = 0; ind < numel(); ind++) {
                    at_index_nonconst(ind) = from.at_index_const(ind);
                }
            } else {
                parallel_for_each(numel(), [this, &from](size_t ind) {
                    at_index_nonconst(ind) = from.at_index_const(ind);
                }, _parallel_batch);
            }
        }
    };





    // ext base/impl
    template <class CategoryT> using ts_ext_inherit = ts_assign_impl<CategoryT>;
    template <class CategoryT>
    class ts_ext : public ts_ext_inherit<CategoryT> {};
    template <class CategoryT>
    class ts_ext_impl : public ts_ext<CategoryT> {};

    // storage
    template <class CategoryT> 
    using ts_storage_inherit = ts_ext_impl<CategoryT>;


    namespace details {
        template <class T>
        struct _static_shape_in_category {
            static constexpr bool value = false;
        };
        template <class ShapeT, class DPT>
        struct _static_shape_in_category<ts_category<ShapeT, DPT>> {
            static constexpr bool value = ShapeT::is_static;
        };
    }

    template <class CategoryT, 
        bool ShapeIsStatic = details::_static_shape_in_category<CategoryT>::value> class ts_storage;
    template <class ShapeT, class DataProviderT>
    class ts_storage<ts_category<ShapeT, DataProviderT>, true>
        : public ts_storage_inherit<ts_category<ShapeT, DataProviderT>> {
    public:
        template <class ... ArgTs>
        constexpr explicit ts_storage(const ShapeT & s, ArgTs && ... args)
            : _data_provider(forward<ArgTs>(args) ...) {}
    public:
        constexpr ShapeT shape_impl() const { return ShapeT(); }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }
    private:
        DataProviderT _data_provider;
    };
    template <class ShapeT, class DataProviderT>
    class ts_storage<ts_category<ShapeT, DataProviderT>, false>
        : public ts_storage_inherit<ts_category<ShapeT, DataProviderT>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
    public:
        template <class ... ArgTs>
        constexpr explicit ts_storage(const ShapeT & s, ArgTs && ... args) 
            : _shape(s), _data_provider(forward<ArgTs>(args) ...) {}
    public:
        constexpr const ShapeT & shape_impl() const { return _shape; }
        ShapeT & shape_impl() { return _shape; }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }
    private:
        ShapeT _shape;
        DataProviderT _data_provider;
    };

    template <class CategoryT> 
    using ts_category_inherit = ts_storage<CategoryT>;


}