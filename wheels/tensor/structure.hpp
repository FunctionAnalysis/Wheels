#pragma once

#include <cassert>

#include "../core/types.hpp"
#include "../core/overloads.hpp"
#include "../core/parallel.hpp"

#include "shape.hpp"
#include "iterators.hpp"

namespace wheels {

    template <class ShapeT, class DataProviderT> 
    class tensor_category;


    // base
    template <class CategoryT>
    class tensor_base {
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
    template <class CategoryT> using tensor_readable_base = tensor_base<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct readable_at_index : no {};
        template <class CategoryT>
        struct readable_at_subs : no {};
        template <class CategoryT>
        struct readable : const_bool<readable_at_index<CategoryT>::value || readable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleReadableAtIndex = tensor_traits::readable_at_index<CategoryT>::value,
        bool EleReadableAtSubs = tensor_traits::readable_at_subs<CategoryT>::value,
        bool EleReadable = EleReadableAtIndex || EleReadableAtSubs>
    class tensor_readable : public tensor_readable_base<CategoryT> {
    public:
        template <class IndexT>
        constexpr int at_index_const(const IndexT & index) const {
            static_assert(always<bool, false, IndexT>::value,
                "the data prodiver is not readable, "
                "explicit instantate tensor_traits::readable_at_index<CategoryT>/tensor_traits::readable_at_subs<CategoryT> and "
                "implement tensor_traits::at_index_const_impl(...)/tensor_traits::at_subs_const_impl(...) to fix this");
        }
        template <class ... SubTs>
        constexpr int at_subs_const(const SubTs & ... subs) const {
            static_assert(always<bool, false, SubTs ...>::value,
                "the data prodiver is not readable, "
                "explicit instantate tensor_traits::readable_at_index<CategoryT>/tensor_traits::readable_at_subs<CategoryT> and "
                "implement tensor_traits::at_index_const_impl(...)/tensor_traits::at_subs_const_impl(...) to fix this");
        }
    };
    template <class CategoryT>
    class tensor_readable<CategoryT, true, true> : public tensor_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return tensor_traits::at_index_const_impl(category(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_subs_const_impl(category(), subs ...);
        }
    };
    template <class CategoryT>
    class tensor_readable<CategoryT, true, false> : public tensor_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return tensor_traits::at_index_const_impl(category(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_index_const_impl(category(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class tensor_readable<CategoryT, false, true> : public tensor_readable_base<CategoryT> {
    public:
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & ... subs) {
                return tensor_traits::at_subs_const_impl(category(), subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_subs_const_impl(category(), subs ...);
        }
    };

    





    // element writable
    template <class CategoryT> using tensor_writable_base = tensor_readable<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct writable_at_index : no {};
        template <class CategoryT>
        struct writable_at_subs : no {};
        template <class CategoryT>
        struct writable : const_bool<writable_at_index<CategoryT>::value || writable_at_subs<CategoryT>::value> {};
    }

    template <class CategoryT, 
        bool EleWritableAtIndex = tensor_traits::writable_at_index<CategoryT>::value, 
        bool EleWritableAtSubs = tensor_traits::writable_at_subs<CategoryT>::value, 
        bool EleWritable = EleWritableAtIndex || EleWritableAtSubs>
    class tensor_writable : public tensor_writable_base<CategoryT> {
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
    template <class CategoryT> class tensor_writer;
    template <class CategoryT>
    class tensor_writable<CategoryT, true, true> : public tensor_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return tensor_traits::at_index_nonconst_impl(category(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_subs_nonconst_impl(category(), subs ...);
        }
    };
    template <class CategoryT>
    class tensor_writable<CategoryT, true, false> : public tensor_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return tensor_traits::at_index_nonconst_impl(category(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using checks_t = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(checks_t::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_index_nonconst_impl(category(), sub2ind(shape(), subs ...));
        }
    };
    template <class CategoryT>
    class tensor_writable<CategoryT, false, true> : public tensor_writable_base<CategoryT> {
    public:
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return tensor_traits::at_subs_nonconst_impl(category(), subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            using checks_t = const_ints<bool, is_int<SubTs>::value ...>;
            static_assert(checks_t::all_v,
                "at_subs(...) requires all subs should be integral or const_ints");
            return tensor_traits::at_subs_nonconst_impl(category(), subs ...);
        }
    };
    






    // brackets and parenthese behavior
    namespace index_tags {
        constexpr auto first = const_index<0>();
        constexpr auto length = const_symbol<0>();
        constexpr auto last = length - const_index<1>();
    }

    namespace details {
        template <class E, class SizeT, class = std::enable_if_t<!is_int<E>::value>>
        constexpr auto _eval_const_expr(const E & e, const SizeT & sz) {
            return e(sz);
        }
        template <class T, class SizeT, class = std::enable_if_t<is_int<T>::value>, class = void>
        constexpr auto _eval_const_expr(const T & t, const SizeT &) {
            return t;
        }
    }

    template <class CategoryT>
    class tensor_bracketensor_and_parenthese_behavior : public tensor_writable<CategoryT> {
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
    using tensor_const_iteratable_base = tensor_bracketensor_and_parenthese_behavior<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct const_iterator_type {
            using type = std::conditional_t<readable<CategoryT>::value, tensor_const_iterator_naive<CategoryT>, void>;
        };
        template <class CategoryT>
        constexpr auto cbegin_impl(const tensor_base<CategoryT> & t) {
            return tensor_const_iterator_naive<CategoryT>(t.category(), 0);
        }
        template <class CategoryT>
        constexpr auto cend_impl(const tensor_base<CategoryT> & t) {
            return tensor_const_iterator_naive<CategoryT>(t.category(), t.numel());
        }     
    }

    template <class CategoryT, class ConstIterT = typename tensor_traits::const_iterator_type<CategoryT>::type>
    class tensor_const_iteratable : public tensor_const_iteratable_base<CategoryT> {
    public:
        constexpr ConstIterT cbegin() const { return tensor_traits::cbegin_impl(category()); }
        constexpr ConstIterT begin() const { return cbegin(); }
        constexpr ConstIterT cend() const { return tensor_traits::cend_impl(category()); }
        constexpr ConstIterT end() const { return cend(); }
    };
    template <class CategoryT>
    class tensor_const_iteratable<CategoryT, void> : public tensor_const_iteratable_base<CategoryT> {};





    // nonconst iteratable
    template <class CategoryT> 
    using tensor_nonconst_iteratable_base = tensor_const_iteratable<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct nonconst_iterator_type {
            using type = std::conditional_t<writable<CategoryT>::value, tensor_nonconst_iterator_naive<CategoryT>, void>;
        };
        template <class CategoryT>
        auto begin_impl(const tensor_base<CategoryT> & t) {
            return tensor_nonconst_iterator_naive<CategoryT>(t.category(), 0);
        }
        template <class CategoryT>
        auto end_impl(const tensor_base<CategoryT> & t) {
            return tensor_nonconst_iterator_naive<CategoryT>(t.category(), t.numel());
        }
    }

    template <class CategoryT, class NonConstIterT = typename tensor_traits::nonconst_iterator_type<CategoryT>::type>
    class tensor_nonconst_iteratable : public tensor_nonconst_iteratable_base<CategoryT> {
    public:
        NonConstIterT begin() { return tensor_traits::begin_impl(category()); }
        NonConstIterT end() { return tensor_traits::end_impl(category()); }
    };
    template <class CategoryT>
    class tensor_nonconst_iteratable<CategoryT, void> : public tensor_nonconst_iteratable_base<CategoryT> {
    public:
        decltype(auto) begin() { return cbegin(); }
        decltype(auto) end() { return cend(); }
    };





    // nonzero iteratable
    template <class CategoryT>
    using tensor_nonzero_iteratable_base = tensor_const_iteratable<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct nonzero_iterator_type {
            using type = std::conditional_t<
                readable<CategoryT>::value, 
                nonzero_iterator_wrapper<typename const_iterator_type<CategoryT>::type>, 
                void
            >;
        };
        template <class CategoryT, class ConstIterT>
        constexpr auto nzbegin_impl(const tensor_const_iteratable<CategoryT, ConstIterT> & t) {
            return wrap_nonzero_iterator(t.cbegin(), t.cend());
        }
        template <class CategoryT, class ConstIterT>
        constexpr auto nzend_impl(const tensor_const_iteratable<CategoryT, ConstIterT> & t) {
            return wrap_nonzero_iterator(t.cend(), t.cend());
        }
    }

    template <class CategoryT, class NonZeroIterT = typename tensor_traits::nonzero_iterator_type<CategoryT>::type>
    class tensor_nonzero_iteratable : public tensor_nonzero_iteratable_base<CategoryT> {
    public:
        constexpr NonZeroIterT nzbegin() const { return tensor_traits::nzbegin_impl(category()); }
        constexpr NonZeroIterT nzend() const { return tensor_traits::nzend_impl(category()); }
    };
    template <class CategoryT>
    class tensor_nonzero_iteratable<CategoryT, void> : public tensor_nonzero_iteratable_base<CategoryT> {};





    // reduce
    template <class CategoryT>
    using tensor_reducible_base = tensor_nonzero_iteratable<CategoryT>;

    template <class CategoryT,
        class ConstIterT = typename tensor_traits::const_iterator_type<CategoryT>::type>
    class tensor_reducible : public tensor_reducible_base<CategoryT> {
    public:
        decltype(auto) max() const { return *std::max_element(cbegin(), cend()); }
        decltype(auto) min() const { return *std::min_element(cbegin(), cend()); }
        bool all() const {
            for (auto it = cbegin(); it != cend(); ++it) {
                if (*it == types<typename CategoryT::value_type>::zero()) {
                    return false;
                }
            }
            return true;
        }
    };
    template <class CategoryT>
    class tensor_reducible<CategoryT, void> : public tensor_reducible_base<CategoryT> {};

    template <class CategoryT,
        class NonZeroIterT = typename tensor_traits::nonzero_iterator_type<CategoryT>::type>
    class tensor_nonzero_reducible : public tensor_reducible<CategoryT> {
    public:
        bool any() const { return nzbegin() != nzend(); }
        bool none() const { return nzbegin() == nzend(); }
        auto sum() const { return std::accumulate(nzbegin(), nzend(), types<typename CategoryT::value_type>::zero()); }
        auto norm_squared() const {
            typename CategoryT::value_type r = types<typename CategoryT::value_type>::zero();
            for (auto it = nzbegin(); it != nzend(); ++it) {
                auto e = *it;
                r += e * e;
            }
            return r;
        }
        auto norm() const { return std::sqrt(norm_squared()); }
    };
    template <class CategoryT>
    class tensor_nonzero_reducible<CategoryT, void> : public tensor_reducible<CategoryT> {};






    // reshape
    template <class CategoryT>
    using tensor_inplace_reshapable_base = tensor_nonzero_reducible<CategoryT>;

    namespace tensor_traits {
        template <class CategoryT>
        struct inplace_reshapable : no {};        
        template <class ShapeT, class DataProviderT>
        struct inplace_reshapable<tensor_category<ShapeT, DataProviderT>>
            : const_bool<!ShapeT::is_static> {};
    }

    template <class CategoryT, 
        bool InplaceReshapable = tensor_traits::inplace_reshapable<CategoryT>::value>
    class tensor_inplace_reshapable : public tensor_inplace_reshapable_base<CategoryT> {
    public:
        template <class ShapeT> 
        void reshape_inplace(const ShapeT &) {}
    };
    template <class CategoryT>
    class tensor_inplace_reshapable<CategoryT, true> : public tensor_inplace_reshapable_base<CategoryT> {
    public:
        template <class ShapeT> 
        void reshape_inplace(const ShapeT & s) {
            category().shape_impl() = s;
        }
    };





    // assignable
    template <class CategoryT> 
    using tensor_assignable_base = tensor_inplace_reshapable<CategoryT>;
    
    namespace tensor_traits {

        template <class CategoryT>
        struct assignable : writable<CategoryT> {};

        template <class CategoryT>
        void reserve_impl(tensor_base<CategoryT> & t, size_t s){}

        template <class CategoryT1, bool WInd1, bool WInd2, 
            class CategoryT2, bool RInd1, bool RInd2>
        void assign_impl(tensor_writable<CategoryT1, WInd1, WInd2, true> & to,
            const tensor_readable<CategoryT2, RInd1, RInd2, true> & from){
            if (from.numel() < 70000) {
                auto & toc = to.category();
                for (size_t ind = 0; ind < from.numel(); ind++) {
                    auto e = from.at_index_const(ind);
                    to.at_index_nonconst(ind) = e;
                }
            } else {
                parallel_for_each(from.numel(), [&to, &from](size_t ind) {
                    auto e = from.at_index_const(ind);
                    to.at_index_nonconst(ind) = e;
                }, 35000);
            }
        }
    }

    template <class CategoryT, bool Assignable = tensor_traits::assignable<CategoryT>::value>
    class tensor_assignable : public tensor_assignable_base<CategoryT> {};
    template <class CategoryT>
    class tensor_assignable<CategoryT, true> : public tensor_assignable_base<CategoryT> {
    public:
        template <class CategoryT2, bool RInd, bool RSub>
        void assign_from(const tensor_readable<CategoryT2, RInd, RSub, true> & from) {
            static_assert(CategoryT::rank == CategoryT2::rank, "rank mismatch");
            reshape_inplace(from.shape());
            assert(shape() == from.shape());
            tensor_traits::reserve_impl(category(), numel());
            tensor_traits::assign_impl(category(), from);
        }
        template <class CategoryT2, bool RInd, bool RSub>
        CategoryT & operator = (const tensor_readable<CategoryT2, RInd, RSub, true> & from) {
            assign_from(from);
            return category();
        }
    };



    namespace details {
        template <class CategoryT>
        struct _types_in_tensor_category {
            using shape_type = void;
            using data_provider_type = void;
            using value_type = void;
        };
        template <class ShapeT, class DataProviderT>
        struct _types_in_tensor_category<tensor_category<ShapeT, DataProviderT>> {
            using shape_type = ShapeT;
            using data_provider_type = DataProviderT;
            using value_type = typename DataProviderT::value_type;
        };
    }


    // specific shape
    template <class CategoryT>
    using tensor_specific_shape_base = tensor_assignable<CategoryT>;
    template <class CategoryT, 
        class ShapeT = typename details::_types_in_tensor_category<CategoryT>::shape_type>
    class tensor_specific_shape : public tensor_specific_shape_base<CategoryT> {};

    // specific value_type
    template <class CategoryT>
    using tensor_specific_value_type_base = tensor_specific_shape<CategoryT>;
    template <class CategoryT, 
        class ValueT = typename details::_types_in_tensor_category<CategoryT>::value_type>
    class tensor_specific_value_type : public tensor_specific_value_type_base<CategoryT> {};




    // storage
    template <class CategoryT> 
    using tensor_storage_base = tensor_specific_value_type<CategoryT>;
    
    template <class CategoryT, 
        bool InPlaceReshapable = tensor_traits::inplace_reshapable<CategoryT>::value> 
    class tensor_storage {};


    namespace details { struct _with_args {};  }
    constexpr details::_with_args with_args;

    
    template <class ShapeT, class DataProviderT>
    class tensor_storage<tensor_category<ShapeT, DataProviderT>, false> 
        : public tensor_storage_base<tensor_category<ShapeT, DataProviderT>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;

        constexpr tensor_storage() {}
        template <class DPT>
        constexpr tensor_storage(const ShapeT & s, DPT && dp)
            : _data_provider(forward<DPT>(dp)) {}
        template <class ... ArgTs>
        constexpr explicit tensor_storage(const details::_with_args &, const ShapeT & s, ArgTs && ... args)
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
    class tensor_storage<tensor_category<ShapeT, DataProviderT>, true>
        : public tensor_storage_base<tensor_category<ShapeT, DataProviderT>> {
    public:
        constexpr tensor_storage() {}
        template <class DPT>
        constexpr tensor_storage(const ShapeT & s, DPT && dp)
            : _shape(s), _data_provider(forward<DPT>(dp)) {}
        template <class ... ArgTs>
        constexpr explicit tensor_storage(const details::_with_args &, const ShapeT & s, ArgTs && ... args)
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


    // category
    template <class CategoryT> 
    using tensor_category_base = tensor_storage<CategoryT>;


#define WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION \
    template <class DPT> \
    constexpr tensor_category(const shape_type & s, DPT && dpt) \
        : tensor_category_base<tensor_category>(s, forward<DPT>(dpt)) {} \
    template <class CategoryT, bool RInd, bool RSub> \
    constexpr tensor_category(const tensor_readable<CategoryT, RInd, RSub, true> & t) { \
        assign_from(t); \
    } \
    constexpr tensor_category(const tensor_category &) = default; \
    tensor_category(tensor_category &&) = default; \
    tensor_category & operator = (const tensor_category &) = default; \
    tensor_category & operator = (tensor_category &&) = default;


    template <class ShapeT, class DataProviderT>
    class tensor_category : public tensor_category_base<tensor_category<ShapeT, DataProviderT>> {
    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename data_provider_type::value_type;
        static constexpr size_t rank = ShapeT::rank;
        WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION
    };

    template <class ShapeT, class DPT>
    constexpr tensor_category<ShapeT, std::decay_t<DPT>> make_tensor(const ShapeT & s, DPT && dp) {
        return tensor_category<ShapeT, std::decay_t<DPT>>(s, forward<DPT>(dp));
    }


}