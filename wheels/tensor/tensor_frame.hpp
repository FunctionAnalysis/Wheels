#pragma once

#include <cassert>

#include "../core/const_expr.hpp"
#include "../core/parallel.hpp"

#include "tensor_shape.hpp"
#include "tensor_utility.hpp"

namespace wheels {

    template <class ShapeT, class DataProviderT> class ts_category;


    // base
    template <class CategoryT>
    class ts_base {
    public:
        // category
        constexpr const CategoryT & category() const { return static_cast<const CategoryT &>(*this); }
        CategoryT & category() { return (CategoryT &)(*this); }

        // shape()
        constexpr decltype(auto) shape() const { return category().shape_impl(); }
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
        constexpr decltype(auto) data_provider() const { return category().data_provider_impl(); }
        decltype(auto) data_provider() { return category().data_provider_impl(); }
    };




    // element readable
    template <class CategoryT> using ts_readable_base = ts_base<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct readable_at_index : no {};
        template <class CategoryT>
        struct readable_at_subs : no {};
        template <class CategoryT>
        struct readable : const_bool<readable_at_index<CategoryT>::value || readable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleReadableAtIndex = ts_props::readable_at_index<CategoryT>::value,
        bool EleReadableAtSubs = ts_props::readable_at_subs<CategoryT>::value,
        bool EleReadable = EleReadableAtIndex || EleReadableAtSubs>
    class ts_readable : public ts_readable_base<CategoryT> {
    public:
        template <class IndexT>
        constexpr int at_index_const(const IndexT & index) const {
            static_assert(always<bool, false, IndexT>::value,
                "the data prodiver is not readable, "
                "explicit instantate ts_props::readable_at_index<CategoryT>/ts_props::readable_at_subs<CategoryT> and "
                "implement ts_props::at_index_const_impl(...)/ts_props::at_subs_const_impl(...) to fix this");
        }
        template <class ... SubTs>
        constexpr int at_subs_const(const SubTs & ... subs) const {
            static_assert(always<bool, false, SubTs ...>::value,
                "the data prodiver is not readable, "
                "explicit instantate ts_props::readable_at_index<CategoryT>/ts_props::readable_at_subs<CategoryT> and "
                "implement ts_props::at_index_const_impl(...)/ts_props::at_subs_const_impl(...) to fix this");
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, true, true> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return ts_props::at_index_const_impl(data_provider(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_subs_const_impl(data_provider(), subs ...);
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, true, false> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return ts_props::at_index_const_impl(data_provider(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_index_const_impl(data_provider(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, false, true> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & ... subs) {
                return ts_props::at_subs_const_impl(data_provider(), subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_subs_const_impl(data_provider(), subs ...);
        }
    };

    





    // element writable
    template <class CategoryT> using ts_writable_base = ts_readable<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct writable_at_index : no {};
        template <class CategoryT>
        struct writable_at_subs : no {};
        template <class CategoryT>
        struct writable : const_bool<writable_at_index<CategoryT>::value || writable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleWritableAtIndex = ts_props::writable_at_index<CategoryT>::value, 
        bool EleWritableAtSubs = ts_props::writable_at_subs<CategoryT>::value, 
        bool EleWritable = EleWritableAtIndex || EleWritableAtSubs>
    class ts_writable : public ts_writable_base<CategoryT> {
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
    template <class CategoryT> class ts_writer;
    template <class CategoryT>
    class ts_writable<CategoryT, true, true> : public ts_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return ts_props::at_index_nonconst_impl(data_provider(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_subs_nonconst_impl(data_provider(), subs ...);
        }
    };
    template <class CategoryT>
    class ts_writable<CategoryT, true, false> : public ts_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return ts_props::at_index_nonconst_impl(data_provider(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_index_nonconst_impl(data_provider(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class ts_writable<CategoryT, false, true> : public ts_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return ts_props::at_subs_nonconst_impl(data_provider(), subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_props::at_subs_nonconst_impl(data_provider(), subs ...);
        }
    };
    






    // brackets and parenthese behavior
    template <class CategoryT>
    class ts_brackets_and_parenthese_behavior : public ts_writable<CategoryT> {
    public:
        // [...] based on at_index
        template <class E>
        constexpr decltype(auto) operator[](const E & e) const {
            return at_index_const(details::_eval_const_expr(e, numel()));
        }
        template <class E>
        decltype(auto) operator[](const E & e) {
            return at_index_nonconst(details::_eval_const_expr(e, numel()));
        }

        // (...) based on at_subs
        template <class ... SubEs>
        constexpr decltype(auto) operator()(const SubEs & ... subes) const {
            return _parenthesis_seq(make_const_sequence(const_int<sizeof...(SubEs)>()), subes ...);
        }
        template <class ... SubEs>
        decltype(auto) operator()(const SubEs & ... subes) {
            return _parenthesis_seq(make_const_sequence(const_int<sizeof...(SubEs)>()), subes ...);
        }

    private:
        template <class ... SubEs, int ... Is>
        constexpr decltype(auto) _parenthesis_seq(const_ints<int, Is...>, const SubEs & ... subes) const {
            return at_subs_const(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
        }
        template <class ... SubEs, int ... Is>
        decltype(auto) _parenthesis_seq(const_ints<int, Is...>, const SubEs & ... subes) {
            return at_subs_nonconst(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
        }
    };





    // const iteratable
    template <class CategoryT> 
    using ts_const_iteratable_base = ts_brackets_and_parenthese_behavior<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct const_iterator_type {
            using type = std::conditional_t<readable<CategoryT>::value, ts_const_iterator_naive<CategoryT>, void>;
        };
        template <class CategoryT>
        constexpr auto cbegin_impl(const ts_base<CategoryT> & t) {
            return ts_const_iterator_naive<CategoryT>(t.category(), 0);
        }
        template <class CategoryT>
        constexpr auto cend_impl(const ts_base<CategoryT> & t) {
            return ts_const_iterator_naive<CategoryT>(t.category(), t.numel());
        }
    }

    template <class CategoryT, class ConstIterT = typename ts_props::const_iterator_type<CategoryT>::type>
    class ts_const_iteratable : public ts_const_iteratable_base<CategoryT> {
    public:
        constexpr ConstIterT cbegin() const { return ts_props::cbegin_impl(category()); }
        constexpr ConstIterT begin() const { return cbegin(); }
        constexpr ConstIterT cend() const { return ts_props::cend_impl(category()); }
        constexpr ConstIterT end() const { return cend(); }
    };
    template <class CategoryT>
    class ts_const_iteratable<CategoryT, void> : public ts_const_iteratable_base<CategoryT> {};





    // nonconst iteratable
    template <class CategoryT> 
    using ts_nonconst_iteratable_base = ts_const_iteratable<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct nonconst_iterator_type {
            using type = std::conditional_t<writable<CategoryT>::value, ts_nonconst_iterator_naive<CategoryT>, void>;
        };
        template <class CategoryT>
        auto begin_impl(const ts_base<CategoryT> & t) {
            return ts_nonconst_iterator_naive<CategoryT>(t.category(), 0);
        }
        template <class CategoryT>
        auto end_impl(const ts_base<CategoryT> & t) {
            return ts_nonconst_iterator_naive<CategoryT>(t.category(), t.numel());
        }
    }

    template <class CategoryT, class NonConstIterT = typename ts_props::nonconst_iterator_type<CategoryT>::type>
    class ts_nonconst_iteratable : public ts_nonconst_iteratable_base<CategoryT> {
    public:
        NonConstIterT begin() { return ts_props::begin_impl(category()); }
        NonConstIterT end() { return ts_props::end_impl(category()); }
    };
    template <class CategoryT>
    class ts_nonconst_iteratable<CategoryT, void> : public ts_nonconst_iteratable_base<CategoryT> {
    public:
        decltype(auto) begin() { return cbegin(); }
        decltype(auto) end() { return cend(); }
    };





    // nonzero iteratable
    template <class CategoryT>
    using ts_nonzero_iteratable_base = ts_const_iteratable<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct nonzero_iterator_type {
            using type = std::conditional_t<readable<CategoryT>::value, ts_nonzero_iterator_naive<CategoryT>, void>;
        };
        template <class CategoryT>
        auto nzbegin_impl(const ts_base<CategoryT> & t) {
            return ts_nonzero_iterator_naive<CategoryT>(t.category(), 0);
        }
        template <class CategoryT>
        auto nzend_impl(const ts_base<CategoryT> & t) {
            return ts_nonzero_iterator_naive<CategoryT>(t.category(), t.numel());
        }
    }

    template <class CategoryT, class NonZeroIterT = typename ts_props::nonzero_iterator_type<CategoryT>::type>
    class ts_nonzero_iteratable : public ts_nonzero_iteratable_base<CategoryT> {
    public:
        NonZeroIterT nzbegin() { return ts_props::nzbegin_impl(category()); }
        NonZeroIterT nzend() { return ts_props::nzend_impl(category()); }
    };
    template <class CategoryT>
    class ts_nonzero_iteratable<CategoryT, void> : public ts_nonzero_iteratable_base<CategoryT> {};





    // reshape
    template <class CategoryT>
    using ts_inplace_reshapable_base = ts_nonzero_iteratable<CategoryT>;

    namespace ts_props {
        template <class CategoryT>
        struct inplace_reshapable : no {};        
        template <class ShapeT, class DataProviderT>
        struct inplace_reshapable<ts_category<ShapeT, DataProviderT>>
            : const_bool<ShapeT::is_static> {};
    }

    template <class CategoryT, 
        bool InplaceReshapable = ts_props::inplace_reshapable<CategoryT>::value>
    class ts_inplace_reshapable : public ts_inplace_reshapable_base<CategoryT> {
    public:
        template <class ShapeT> 
        void reshape_inplace(const ShapeT &) {}
    };
    template <class CategoryT>
    class ts_inplace_reshapable<CategoryT, true> : public ts_inplace_reshapable_base<CategoryT> {
    public:
        template <class ShapeT> 
        void reshape_inplace(const ShapeT & s) {
            category().shape_impl() = s;
        }
    };





    // assignable
    template <class CategoryT> 
    using ts_assignable_base = ts_inplace_reshapable<CategoryT>;
    
    namespace ts_props {

        template <class CategoryT>
        struct assignable : writable<CategoryT> {};

        template <class CategoryT>
        void reserve_impl(ts_base<CategoryT> & t, size_t s){}

        static constexpr size_t _parallel_thres = 70000;
        static constexpr size_t _parallel_batch = 35000;

        template <class CategoryT1, bool WInd1, bool WInd2, 
            class CategoryT2, bool RInd1, bool RInd2>
        void assign_impl(ts_writable<CategoryT1, WInd1, WInd2, true> & to,
            const ts_readable<CategoryT2, RInd1, RInd2, true> & from){
            if (numel() < _parallel_thres) {
                for (size_t ind = 0; ind < numel(); ind++) {
                    to.at_index_nonconst(ind) = from.at_index_const(ind);
                }
            } else {
                parallel_for_each(numel(), [&to, &from](size_t ind) {
                    to.at_index_nonconst(ind) = from.at_index_const(ind);
                }, _parallel_batch);
            }
        }
    }

    template <class CategoryT, bool Assignable = ts_props::assignable<CategoryT>::value>
    class ts_assignable : public ts_assignable_base<CategoryT> {};
    template <class CategoryT>
    class ts_assignable<CategoryT, true> : public ts_assignable_base<CategoryT> {
    public:
        template <class CategoryT2, bool RInd, bool RSub>
        void assign_from(const ts_readable<CategoryT2, RInd, RSub, true> & from) {
            reshape_inplace(from.shape());
            assert(shape() == to.shape());
            ts_props::reserve_impl(*this, numel());
            ts_props::assign_impl(*this, from);
        }
        template <class CategoryT2, bool RInd, bool RSub>
        CategoryT & operator = (const ts_readable<CategoryT2, RInd, RSub, true> & from) {
            assign_from(from);
            return category();
        }
    };





    // extensions
    template <class CategoryT> 
    using ts_extensions_base = ts_assignable<CategoryT>;
    template <class CategoryT>
    class ts_extensions : public ts_extensions_base<CategoryT> {};




    // storage
    template <class CategoryT> 
    using ts_storage_base = ts_extensions<CategoryT>;
    
    template <class CategoryT, 
        bool InPlaceReshapable = ts_props::inplace_reshapable<CategoryT>::value> 
    class ts_storage {};


    struct _with_args {};
    constexpr _with_args with_args;

    
    template <class ShapeT, class DataProviderT>
    class ts_storage<ts_category<ShapeT, DataProviderT>, false> 
        : public ts_storage_base<ts_category<ShapeT, DataProviderT>> {
    public:
        template <class DPT>
        constexpr ts_storage(const ShapeT & s, DPT && dp)
            : _data_provider(forward<DPT>(dp)) {}
        template <class ... ArgTs>
        constexpr explicit ts_storage(const _with_args &, const ShapeT & s, ArgTs && ... args)
            : _data_provider(forward<ArgTs>(args) ...) {}
    public:
        constexpr ShapeT shape_impl() const { return ShapeT(); }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }
    private:
        DataProviderT _data_provider;
    };

    template <class ShapeT, class DataProviderT>
    class ts_storage<ts_category<ShapeT, DataProviderT>, true>
        : public ts_storage_base<ts_category<ShapeT, DataProviderT>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
    public:
        template <class DPT>
        constexpr ts_storage(const ShapeT & s, DPT && dp)
            : _shape(s), _data_provider(forward<DPT>(dp)) {}
        template <class ... ArgTs>
        constexpr explicit ts_storage(const _with_args &, const ShapeT & s, ArgTs && ... args)
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
    using ts_category_base = ts_storage<CategoryT>;







}