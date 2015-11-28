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
            static_assert(Idx >= 0 && Idx < CategoryT::rank, "invalid Idx");
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

    namespace ts_traits {
        template <class CategoryT>
        struct readable_at_index : no {};
        template <class CategoryT>
        struct readable_at_subs : no {};
        template <class CategoryT>
        struct readable : const_bool<readable_at_index<CategoryT>::value || readable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleReadableAtIndex = ts_traits::readable_at_index<CategoryT>::value,
        bool EleReadableAtSubs = ts_traits::readable_at_subs<CategoryT>::value,
        bool EleReadable = EleReadableAtIndex || EleReadableAtSubs>
    class ts_readable : public ts_readable_base<CategoryT> {
    public:
        template <class IndexT>
        constexpr int at_index_const(const IndexT & index) const {
            static_assert(always<bool, false, IndexT>::value,
                "the data prodiver is not readable, "
                "explicit instantate ts_traits::readable_at_index<CategoryT>/ts_traits::readable_at_subs<CategoryT> and "
                "implement ts_traits::at_index_const_impl(...)/ts_traits::at_subs_const_impl(...) to fix this");
        }
        template <class ... SubTs>
        constexpr int at_subs_const(const SubTs & ... subs) const {
            static_assert(always<bool, false, SubTs ...>::value,
                "the data prodiver is not readable, "
                "explicit instantate ts_traits::readable_at_index<CategoryT>/ts_traits::readable_at_subs<CategoryT> and "
                "implement ts_traits::at_index_const_impl(...)/ts_traits::at_subs_const_impl(...) to fix this");
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, true, true> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return ts_traits::at_index_const_impl(category(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_subs_const_impl(category(), subs ...);
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, true, false> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return ts_traits::at_index_const_impl(category(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_index_const_impl(category(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class ts_readable<CategoryT, false, true> : public ts_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & ... subs) {
                return ts_traits::at_subs_const_impl(category(), subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_subs_const_impl(category(), subs ...);
        }
    };

    





    // element writable
    template <class CategoryT> using ts_writable_base = ts_readable<CategoryT>;

    namespace ts_traits {
        template <class CategoryT>
        struct writable_at_index : no {};
        template <class CategoryT>
        struct writable_at_subs : no {};
        template <class CategoryT>
        struct writable : const_bool<writable_at_index<CategoryT>::value || writable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleWritableAtIndex = ts_traits::writable_at_index<CategoryT>::value, 
        bool EleWritableAtSubs = ts_traits::writable_at_subs<CategoryT>::value, 
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
            return ts_traits::at_index_nonconst_impl(category(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_subs_nonconst_impl(category(), subs ...);
        }
    };
    template <class CategoryT>
    class ts_writable<CategoryT, true, false> : public ts_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return ts_traits::at_index_nonconst_impl(category(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_index_nonconst_impl(category(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class ts_writable<CategoryT, false, true> : public ts_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return ts_traits::at_subs_nonconst_impl(category(), subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using _checks = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(_checks::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return ts_traits::at_subs_nonconst_impl(category(), subs ...);
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

    namespace ts_traits {
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

    template <class CategoryT, class ConstIterT = typename ts_traits::const_iterator_type<CategoryT>::type>
    class ts_const_iteratable : public ts_const_iteratable_base<CategoryT> {
    public:
        constexpr ConstIterT cbegin() const { return ts_traits::cbegin_impl(category()); }
        constexpr ConstIterT begin() const { return cbegin(); }
        constexpr ConstIterT cend() const { return ts_traits::cend_impl(category()); }
        constexpr ConstIterT end() const { return cend(); }
    };
    template <class CategoryT>
    class ts_const_iteratable<CategoryT, void> : public ts_const_iteratable_base<CategoryT> {};





    // nonconst iteratable
    template <class CategoryT> 
    using ts_nonconst_iteratable_base = ts_const_iteratable<CategoryT>;

    namespace ts_traits {
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

    template <class CategoryT, class NonConstIterT = typename ts_traits::nonconst_iterator_type<CategoryT>::type>
    class ts_nonconst_iteratable : public ts_nonconst_iteratable_base<CategoryT> {
    public:
        NonConstIterT begin() { return ts_traits::begin_impl(category()); }
        NonConstIterT end() { return ts_traits::end_impl(category()); }
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

    namespace ts_traits {
        template <class CategoryT>
        struct nonzero_iterator_type {
            using type = std::conditional_t<
                readable<CategoryT>::value, 
                nonzero_iterator_of<typename const_iterator_type<CategoryT>::type>, 
                void
            >;
        };
        template <class CategoryT, class ConstIterT>
        constexpr nonzero_iterator_of<ConstIterT> nzbegin_impl(const ts_const_iteratable<CategoryT, ConstIterT> & t) {
            return nonzero_iterator_of<ConstIterT>(t.cbegin(), t.cend());
        }
        template <class CategoryT, class ConstIterT>
        constexpr nonzero_iterator_of<ConstIterT> nzend_impl(const ts_const_iteratable<CategoryT, ConstIterT> & t) {
            return nonzero_iterator_of<ConstIterT>(t.cend(), t.cend());
        }
    }

    template <class CategoryT, class NonZeroIterT = typename ts_traits::nonzero_iterator_type<CategoryT>::type>
    class ts_nonzero_iteratable : public ts_nonzero_iteratable_base<CategoryT> {
    public:
        constexpr NonZeroIterT nzbegin() const { return ts_traits::nzbegin_impl(category()); }
        constexpr NonZeroIterT nzend() const { return ts_traits::nzend_impl(category()); }
    };
    template <class CategoryT>
    class ts_nonzero_iteratable<CategoryT, void> : public ts_nonzero_iteratable_base<CategoryT> {};





    // reshape
    template <class CategoryT>
    using ts_inplace_reshapable_base = ts_nonzero_iteratable<CategoryT>;

    namespace ts_traits {
        template <class CategoryT>
        struct inplace_reshapable : no {};        
        template <class ShapeT, class DataProviderT>
        struct inplace_reshapable<ts_category<ShapeT, DataProviderT>>
            : const_bool<!ShapeT::is_static> {};
    }

    template <class CategoryT, 
        bool InplaceReshapable = ts_traits::inplace_reshapable<CategoryT>::value>
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



    // index_accessible_from_iterator
    // iter2ind
    namespace ts_traits {

        template <class IterT>
        struct index_accessible_from_iterator : no {};

        template <class CategoryT>
        struct index_accessible_from_iterator<ts_const_iterator_naive<CategoryT>> : yes {};
        template <class CategoryT>
        constexpr size_t iter2ind(const ts_const_iterator_naive<CategoryT> & iter) {
            return iter.ind;
        }

        template <class CategoryT>
        struct index_accessible_from_iterator<ts_nonconst_iterator_naive<CategoryT>> : yes {};
        template <class CategoryT>
        constexpr size_t iter2ind(const ts_nonconst_iterator_naive<CategoryT> & iter) {
            return iter.ind;
        }

        template <class IterT>
        struct index_accessible_from_iterator<nonzero_iterator_of<IterT>>
            : index_accessible_from_iterator<IterT> {};
        template <class IterT>
        constexpr size_t iter2ind(const nonzero_iterator_of<IterT> & iter) {
            return iter2ind(iter.iter);
        }

        template <class IterT>
        struct index_accessible_from_iterator<second_in_pair_iterator_of<IterT>> : yes {};
        template <class IterT>
        constexpr size_t iter2ind(const second_in_pair_iterator_of<IterT> & iter) {
            return iter.iter->first;
        }

    }





    // assignable
    template <class CategoryT> 
    using ts_assignable_base = ts_inplace_reshapable<CategoryT>;
    
    namespace ts_traits {

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
            if (from.numel() < _parallel_thres) {
                auto & toc = to.category();
                for (size_t ind = 0; ind < from.numel(); ind++) {
                    auto e = from.at_index_const(ind);
                    to.at_index_nonconst(ind) = e;
                }
            } else {
                parallel_for_each(from.numel(), [&to, &from](size_t ind) {
                    auto e = from.at_index_const(ind);
                    to.at_index_nonconst(ind) = e;
                }, _parallel_batch);
            }
        }
    }

    template <class CategoryT, bool Assignable = ts_traits::assignable<CategoryT>::value>
    class ts_assignable : public ts_assignable_base<CategoryT> {};
    template <class CategoryT>
    class ts_assignable<CategoryT, true> : public ts_assignable_base<CategoryT> {
    public:
        template <class CategoryT2, bool RInd, bool RSub>
        void assign_from(const ts_readable<CategoryT2, RInd, RSub, true> & from) {
            static_assert(CategoryT::rank == CategoryT2::rank, "rank mismatch");
            reshape_inplace(from.shape());
            assert(shape() == from.shape());
            ts_traits::reserve_impl(category(), numel());
            ts_traits::assign_impl(category(), from);
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
        bool InPlaceReshapable = ts_traits::inplace_reshapable<CategoryT>::value> 
    class ts_storage {};


    struct _with_args {};
    constexpr _with_args with_args;

    
    template <class ShapeT, class DataProviderT>
    class ts_storage<ts_category<ShapeT, DataProviderT>, false> 
        : public ts_storage_base<ts_category<ShapeT, DataProviderT>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
        static constexpr size_t rank = ShapeT::rank;

    public:
        constexpr ts_storage() {}
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

        template <class Archiver>
        void serialize(Archiver & ar) { ar(_data_provider); }

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
        static constexpr size_t rank = ShapeT::rank;

    public:
        constexpr ts_storage() {}
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

        template <class Archiver> 
        void serialize(Archiver & ar) { ar(_shape, _data_provider); }

    private:
        ShapeT _shape;
        DataProviderT _data_provider;
    };

    template <class CategoryT> 
    using ts_category_base = ts_storage<CategoryT>;







}