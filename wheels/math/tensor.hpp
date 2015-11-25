#pragma once

#include <cassert>
#include <array>
#include <vector>

#include "../core/macros.hpp"
#include "../core/const_expr.hpp"
#include "../core/types.hpp"
#include "../core/parallel.hpp"

#include "tensor_shape.hpp"
#include "tensor_data.hpp"

namespace wheels {

    // forward declaration

    template <class ShapeT, 
        class DataProviderT,
        bool ShapeIsStatic = ShapeT::is_static>
    class tensor_layout;


    namespace details {
        template <class LayoutT>
        struct _layout_property {};
        template <class ShapeT, class DataProviderT, bool S>
        struct _layout_property<tensor_layout<ShapeT, DataProviderT, S>> {
            static constexpr bool element_readable_at_index = 
                tdp::is_element_readable_at_index<DataProviderT>::value;
            static constexpr bool element_writable_at_index = 
                tdp::is_element_writable_at_index<DataProviderT>::value;
            static constexpr bool element_readable_at_subs = 
                tdp::is_element_readable_at_subs<DataProviderT>::value;
            static constexpr bool element_writable_at_subs = 
                tdp::is_element_writable_at_subs<DataProviderT>::value;
        };
    }

    template <class LayoutT>
    class tensor_method_core;

    template <class LayoutT,
        bool EleReadableAtIndex = details::_layout_property<LayoutT>::element_readable_at_index,
        bool EleReadableAtSubs = details::_layout_property<LayoutT>::element_readable_at_subs>
    class tensor_method_read_element;

    template <class LayoutT,
        bool EleWritableAtIndex = details::_layout_property<LayoutT>::element_writable_at_index,
        bool EleWritableAtSubs = details::_layout_property<LayoutT>::element_writable_at_subs>
    class tensor_method_write_element;

    template <class LayoutT,
        bool EleReadable = details::_layout_property<LayoutT>::element_readable_at_index ||
            details::_layout_property<LayoutT>::element_readable_at_subs>
    class tensor_method_iterate_const;

    template <class LayoutT,
        bool EleWritable = details::_layout_property<LayoutT>::element_writable_at_index ||
            details::_layout_property<LayoutT>::element_writable_at_subs>
    class tensor_method_iterate_nonconst;

    template <class LayoutT,
        bool AssignByIndex = details::_layout_property<LayoutT>::element_readable_at_index>
    class tensor_method_copy;

    template <class LayoutT>
    class tensor_base;



    // tensor_method_core
    template <class LayoutT>
    class tensor_method_core {
    public:
        using layout_type = LayoutT;

        constexpr const layout_type & layout() const { return (const layout_type &)(*this); }
        layout_type & layout() { return (layout_type &)(*this); }

        // shape related
        constexpr decltype(auto) shape() const { return layout().shape_impl(); }
        constexpr auto rank() const { return const_ints<int, decltype(shape())::rank>(); }
        constexpr auto degree_sequence() const { return make_const_sequence(rank()); }

        template <class T, T Idx>
        constexpr auto size(const const_ints<T, Idx> & i) const {
            return shape().at(i);
        }
        auto numel() const { return shape().magnitude(); }    
    };


    // tensor_method_read_element
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, true, true>
        : public tensor_method_core<LayoutT> {
    public:
        static constexpr bool element_readable_at_index = true;
        static constexpr bool element_readable_at_subs = true;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, false, true>
        : public tensor_method_core<LayoutT> {
    public:
        static constexpr bool element_readable_at_index = false;
        static constexpr bool element_readable_at_subs = true;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & ... subs) {
                return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, true, false>
        : public tensor_method_core<LayoutT> {
    public:
        static constexpr bool element_readable_at_index = true;
        static constexpr bool element_readable_at_subs = false;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index_const(const IndexT & index) const {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_const(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_index(layout().data_provider_impl(),
                sub2ind(shape(), subs ...));
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, false, false>
        : public tensor_method_core<LayoutT> {
    public:
        static constexpr bool element_readable_at_index = false;
        static constexpr bool element_readable_at_subs = false;
        template <class IndexT> 
        constexpr int at_index_const(const IndexT & index) const {
            static_assert(always<bool, false, IndexT>::value, "the data prodiver is not readable");
        }
        template <class ... SubTs>
        constexpr int at_subs_const(const SubTs & ... subs) const {
            static_assert(always<bool, false, SubTs ...>::value, "the data prodiver is not readable");
        }
    };


    

    // tensor_method_write_element
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, true, true>
        : public tensor_method_read_element<LayoutT> {
    public:
        static constexpr bool element_writable_at_index = true;
        static constexpr bool element_writable_at_subs = true;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, false, true>
        : public tensor_method_read_element<LayoutT> {
    public:
        static constexpr bool element_writable_at_index = false;
        static constexpr bool element_writable_at_subs = true;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, true, false>
        : public tensor_method_read_element<LayoutT> {
    public:
        static constexpr bool element_writable_at_index = true;
        static constexpr bool element_writable_at_subs = false;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index_nonconst(const IndexT & index) {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs_nonconst(const SubTs & ... subs) {
           /* static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");*/
            return tdp::element_at_index(layout().data_provider_impl(),
                sub2ind(shape(), subs ...));
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, false, false>
        : public tensor_method_read_element<LayoutT> {
    public:
        static constexpr bool element_writable_at_index = false;
        static constexpr bool element_writable_at_subs = false;
        template <class IndexT>
        constexpr decltype(auto) at_index_nonconst(const IndexT & index) const {
            return at_index_const(index);
        }
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_nonconst(const SubTs & ... subs) const {
            return at_subs_const(subs ...);
        }
    };





    // tensor_method_iterate_const
    template <class LayoutT>
    class tensor_method_iterate_const<LayoutT, true>  // readable
        : public tensor_method_write_element<LayoutT> {
        using this_t = tensor_method_iterate_const<LayoutT, true>;
    public:
        struct const_iterator : std::iterator<std::random_access_iterator_tag,
            typename LayoutT::value_type,
            std::ptrdiff_t> {
            const this_t & self;
            size_t ind;
            constexpr const_iterator(const this_t & s, size_t i = 0) : self(s), ind(i) {}
            constexpr decltype(auto) operator * () const { return self.at_index_const(ind); }
            constexpr decltype(auto) operator -> () const { return self.at_index_const(ind); }
            const_iterator & operator ++() { ++ind; return *this; }
            const_iterator & operator --() { assert(ind != 0);  --ind; return *this; }
            const_iterator & operator +=(size_t s) { ind += s; return *this; }
            const_iterator & operator -=(size_t s) { ind -= s; return *this; }
            constexpr const_iterator operator + (size_t s) const { return const_iterator(self, ind + s); }
            constexpr const_iterator operator - (size_t s) const { return const_iterator(self, ind - s); }
            std::ptrdiff_t operator - (const const_iterator & it) const { return ind - it.ind; }
            constexpr bool operator == (const const_iterator & it) const {
                assert(&self == &(it.self)); 
                return ind == it.ind; 
            }
            constexpr bool operator != (const const_iterator & it) const {
                return ind != it.ind;
            }
            constexpr bool operator < (const const_iterator & it) const {
                return ind < it.ind;
            }
        };

        constexpr const_iterator begin() const { return const_iterator(*this, 0); }
        constexpr const_iterator cbegin() const { return const_iterator(*this, 0); }

        constexpr const_iterator end() const { return const_iterator(*this, numel()); }
        constexpr const_iterator cend() const { return const_iterator(*this, numel()); }
    };
    template <class LayoutT>
    class tensor_method_iterate_const<LayoutT, false>  // not readable
        : public tensor_method_write_element<LayoutT> {
    public:
        using const_iterator = void;

        constexpr const_iterator begin() const {}
        constexpr const_iterator cbegin() const {}

        constexpr const_iterator end() const {}
        constexpr const_iterator cend() const {}
    };



    // tensor_method_iterate_nonconst
    template <class LayoutT>
    class tensor_method_iterate_nonconst<LayoutT, true>  // writable
        : public tensor_method_iterate_const<LayoutT> {
        using this_t = tensor_method_iterate_nonconst<LayoutT, true>;
    public:
        struct iterator : std::iterator<std::random_access_iterator_tag, 
            typename LayoutT::value_type,
            std::ptrdiff_t> {
            this_t & self;
            size_t ind;
            constexpr iterator(this_t & s, size_t i = 0) : self(s), ind(i) {}
            decltype(auto) operator * () const { return self.at_index_nonconst(ind); }
            decltype(auto) operator -> () const { return self.at_index_nonconst(ind); }
            iterator & operator ++() { ++ind; return *this; }
            iterator & operator --() { assert(ind != 0);  --ind; return *this; }
            iterator & operator +=(size_t s) { ind += s; return *this; }
            iterator & operator -=(size_t s) { ind -= s; return *this; }
            constexpr iterator operator + (size_t s) const { return iterator(self, ind + s); }
            constexpr iterator operator - (size_t s) const { return iterator(self, ind - s); }
            std::ptrdiff_t operator - (const iterator & it) const { return ind - it.ind; }
            constexpr bool operator == (const iterator & it) const {
                assert(&self == &(it.self));
                return ind == it.ind;
            }
            constexpr bool operator != (const iterator & it) const {
                return ind != it.ind;
            }
            constexpr bool operator < (const iterator & it) const {
                return ind < it.ind;
            }
        };

        iterator begin() { return iterator(*this, 0); }
        iterator end() { return iterator(*this, numel()); }
    };
    template <class LayoutT>
    class tensor_method_iterate_nonconst<LayoutT, false>  // not writable
        : public tensor_method_iterate_const<LayoutT> {
    public:
        using iterator = typename tensor_method_iterate_const<LayoutT>::const_iterator;

        iterator begin() { return cbegin(); }
        iterator end() { return cend(); }
    };





    // tensor_method_reduce
    static constexpr size_t _parallel_thres = 70000;
    static constexpr size_t _parallel_batch = 35000;
    template <class LayoutT>
    class tensor_method_reduce : public tensor_method_iterate_nonconst<LayoutT> {
    public:
        // reduce
        template <class T, class ReduceT>
        T accumulate(const T & base, ReduceT && redux) const {
            T result = base;
            if (numel() < _parallel_thres) {
                for (size_t ind = 0; ind < numel(); ind++) {
                    result = redux(result, at_index_const(ind));
                }
            } else {
                parallel_for_each(numel(), [this, &result, &redux](size_t ind) {
                    result = redux(result, at_index_const(ind));
                }, _parallel_batch);
            }
            return result;
        }
        auto sum() const { return accumulate(typename LayoutT::value_type(0), binary_op_plus()); }
        auto mean() const { return sum() / numel(); }
        auto prod() const { return accumulate(typename LayoutT::value_type(1), binary_op_mul()); }
        bool all() const { return accumulate(true, binary_op_and()); }
        bool any() const { return accumulate(false, binary_op_or()); }
        bool none() const { return !any(); }

        // norm_squared
        auto norm_squared() const {
            typename LayoutT::value_type result = 0;
            if (numel() < _parallel_thres) {
                for (size_t ind = 0; ind < numel(); ind++) {
                    decltype(auto) e = at_index_const(ind);
                    result += e * e;
                }
            } else {
                parallel_for_each(numel(), [this, &result](size_t ind) {
                    decltype(auto) e = at_index_const(ind);
                    result += e * e;
                }, _parallel_batch);
            }
            return result;
        }

        // norm
        auto norm() const { return std::sqrt(norm_squared()); }
    };







    // tensor_method_copy
    template <class LayoutT>
    class tensor_method_copy<LayoutT, true> // by index
        : public tensor_method_reduce<LayoutT> {
    public:
        static constexpr bool copy_by_index = true;
        // copy_to
        template <class LayoutToT, bool B1, bool B2>
        void copy_to(tensor_method_write_element<LayoutToT, B1, B2> & to) const {
            static_assert(B1 || B2, "'to' is not writable");
            assert(shape() == to.shape());
            tdp::reserve_storage(to.shape(), to.layout().data_provider_impl());
            if (numel() < _parallel_thres) {
                for (int ind = 0; ind < numel(); ind++) {
                    to.at_index_nonconst(ind) = at_index_const(ind);
                }
            } else {
                parallel_for_each(numel(), [this, &to](size_t ind) {
                    to.at_index_nonconst(ind) = at_index_const(ind);
                }, _parallel_batch);
            }
        }
    };
    template <class LayoutT>
    class tensor_method_copy<LayoutT, false> // by subs
        : public tensor_method_reduce<LayoutT> {
    public:
        static constexpr bool copy_by_index = false;
        // copy_to
        template <class LayoutToT, bool B1, bool B2>
        void copy_to(tensor_method_write_element<LayoutToT, B1, B2> & to) const {
            static_assert(B1 || B2, "'to' is not writable");
            assert(shape() == to.shape());
            tdp::reserve_storage(to.shape(), to.layout().data_provider_impl());
            if (numel() < _parallel_thres) {
                for_each_subscript(shape(), [this, &to](const auto & ... subs) {
                    to.at_subs_nonconst(subs ...) = at_subs_const(subs ...);
                });
            } else {
                parallel_for_each(numel(), [this, &to](size_t ind) {
                    invoke_with_subs(shape(), ind, [this, &to](const auto & ... subs) {
                        to.at_subs_nonconst(subs ...) = at_subs_const(subs ...);
                    });
                }, _parallel_batch);
            }
        }
    };



    namespace index_tags {
        constexpr auto first = const_index<0>();
        constexpr auto length = const_symbol<0>();
        constexpr auto last = length - const_index<1>();
    }

    namespace details {
        template <class E, class SizeT, class = std::enable_if_t<is_const_expr<E>::value>>
        constexpr auto _eval_const_expr(const E & e, const SizeT & sz) {
            return e(sz);
        }
        template <class T, class SizeT, class = std::enable_if_t<!is_const_expr<T>::value>, class = void>
        constexpr auto _eval_const_expr(const T & t, const SizeT &) {
            return t;
        }
    }




    // tensor_base
    template <class LayoutT>
    class tensor_base : public tensor_method_copy<LayoutT> {
    public:
        using layout_type = LayoutT;

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



    // tensor_extended
    // overload to implement more methods
    template <class LayoutT> 
    class tensor_extended 
        : public tensor_base<LayoutT> {
    };




    namespace details {
        template <class Name> 
        struct _tensor_construct_tag {
            constexpr _tensor_construct_tag() {}
        };
        template <class T> struct _is_tensor_construct_tag : no {};
        template <class T> struct _is_tensor_construct_tag<_tensor_construct_tag<T>> : yes {};
    }
    struct _with_elements {};
    constexpr details::_tensor_construct_tag<_with_elements> with_elements;
    struct _with_args {};
    constexpr details::_tensor_construct_tag<_with_args> with_args;


    // tensor_layout with non-static shape
    template <class ShapeT, class DataProviderT>
    class tensor_layout<ShapeT, DataProviderT, false> 
        : public tensor_extended<tensor_layout<ShapeT, DataProviderT, false>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = false;

        template <class ST, class DPT>
        friend constexpr tensor_layout<ST, std::decay_t<DPT>>
            compose_tensor(const ST & shape, DPT && dp);

    public:  
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT; 
        using value_type = typename DataProviderT::value_type;
        
    public:
        template <wheels_enable_if(tdp::is_default_constructible<data_provider_type>::value)>
        constexpr tensor_layout() {}

        template <wheels_enable_if(tdp::is_copy_constructible<data_provider_type>::value)>
        constexpr tensor_layout(const ShapeT & s, const DataProviderT & stg)
            : _shape(s), _data_provider(stg) {}

        template <wheels_enable_if(tdp::is_move_constructible<data_provider_type>::value)>
        constexpr tensor_layout(const ShapeT & s, DataProviderT && stg)
            : _shape(s), _data_provider(std::move(stg)) {}

        // tensor_layout(shape)
        template <wheels_enable_if(tdp::is_constructible_with_shape<data_provider_type>::value)>
        constexpr explicit tensor_layout(const ShapeT & s)
            : _shape(s), _data_provider(tdp::construct_with_shape(types<data_provider_type>(), s)) {}

        // tensor_layout(shape, with_elements, elements ...)
        template <wheels_enable_if(tdp::is_constructible_with_shape_elements<data_provider_type>::value),
            class ... EleTs>
        constexpr tensor_layout(const ShapeT & s, details::_tensor_construct_tag<_with_elements>, EleTs && ... eles)
            : _shape(s), _data_provider(tdp::construct_with_shape_elements(types<data_provider_type>(), _shape, forward<EleTs>(eles) ...)) {}

        // tensor_layout(shape, with_args, args ...)
        template <class ... ArgTs>
        constexpr tensor_layout(const ShapeT & s, details::_tensor_construct_tag<_with_args>, ArgTs && ... args)
            : _shape(s), _data_provider(tdp::construct_with_args(types<data_provider_type>(), forward<ArgTs>(args) ...)) {}

        // default copy
        constexpr tensor_layout(const tensor_layout &) = default;
        constexpr tensor_layout(tensor_layout &&) = default;
        tensor_layout & operator = (const tensor_layout &) = default;
        tensor_layout & operator = (tensor_layout &&) = default;

        // copy
        template <class ShapeFromT, class DataProviderFromT, bool B>
        constexpr tensor_layout(const tensor_layout<ShapeFromT, DataProviderFromT, B> & from) 
            : _shape(from.shape()) {
            from.copy_to(*this);
        }
        template <class ShapeFromT, class DataProviderFromT, bool B>
        tensor_layout & operator = (const tensor_layout<ShapeFromT, DataProviderFromT, B> & from) {
            _shape = from.shape();
            from.copy_to(*this);
            return *this;
        }

        // interfaces
        constexpr const ShapeT & shape_impl() const { return _shape; }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }

        template <class Archiver>
        void serialize(Archiver & ar) {
            ar(_shape, _data_provider);
        }

    private:
        template <class DPT>
        constexpr tensor_layout(const ShapeT & shape, DPT && dp) 
            : _shape(shape), _data_provider(forward<DPT>(dp)) {}

    private:
        ShapeT _shape;
        DataProviderT _data_provider;
    };


    // tensor_layout with static shape
    template <class ShapeT, class DataProviderT>
    class tensor_layout<ShapeT, DataProviderT, true> 
        : public tensor_extended<tensor_layout<ShapeT, DataProviderT, true>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = true;

        template <class ST, class DPT>
        friend constexpr tensor_layout<ST, std::decay_t<DPT>>
            compose_tensor(const ST & shape, DPT && dp);

    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;
        using value_type = typename DataProviderT::value_type;

    public:
        template <wheels_enable_if(tdp::is_constructible_with_shape<data_provider_type>::value)>
        constexpr tensor_layout() : _data_provider(tdp::construct_with_shape(types<data_provider_type>(), ShapeT())) {}

        // tensor_layout(with_elements, elements ...)
        template <wheels_enable_if(tdp::is_constructible_with_shape_elements<data_provider_type>::value),
            class ... EleTs>
        constexpr tensor_layout(const details::_tensor_construct_tag<_with_elements> &, EleTs && ... eles)
            : _data_provider(tdp::construct_with_shape_elements(types<data_provider_type>(), ShapeT(), forward<EleTs>(eles) ...)) {}

        template <class ... ArgTs>
        constexpr tensor_layout(details::_tensor_construct_tag<_with_args>, ArgTs && ... args)
            : _data_provider(tdp::construct_with_args(types<data_provider_type>(), forward<ArgTs>(args) ...)) {}

        // default copy/assign
        constexpr tensor_layout(const tensor_layout &) = default;
        constexpr tensor_layout(tensor_layout &&) = default;
        tensor_layout & operator = (const tensor_layout &) = default;
        tensor_layout & operator = (tensor_layout &&) = default;

        // copy
        template <class ShapeFromT, class DataProviderFromT, bool B>
        constexpr tensor_layout(const tensor_layout<ShapeFromT, DataProviderFromT, B> & from) {
            from.copy_to(*this);
        }
        template <class ShapeFromT, class DataProviderFromT, bool B>
        tensor_layout & operator = (const tensor_layout<ShapeFromT, DataProviderFromT, B> & from) {
            from.copy_to(*this);
            return *this;
        }

        // interfaces
        constexpr ShapeT shape_impl() const { return ShapeT(); }
        constexpr const DataProviderT & data_provider_impl() const { return _data_provider; }
        DataProviderT & data_provider_impl() { return _data_provider; }

        template <class Archiver>
        void serialize(Archiver & ar) {
            ar(_data_provider);
        }

    private:
        template <class DPT>
        constexpr tensor_layout(const ShapeT & shape, DPT && dp)
            : _data_provider(forward<DPT>(dp)) {}

    private:
        DataProviderT _data_provider;
    };



    // is_tensor
    template <class T> struct is_tensor : no {};
    template <class ShapeT, class DataProviderT, bool B>
    struct is_tensor<tensor_layout<ShapeT, DataProviderT, B>> : yes {};


    // compose_tensor
    template <class ST, class DPT>
    constexpr tensor_layout<ST, std::decay_t<DPT>> compose_tensor(const ST & shape, DPT && dp) {
        return tensor_layout<ST, std::decay_t<DPT>>(shape, forward<DPT>(dp));
    }




    namespace details {
        template <class T, class ShapeT, bool ShapeIsStatic = ShapeT::is_static>
        struct _recommend_data_provider {
            using type = std::vector<T>;
        };
        template <class T, class ShapeT>
        struct _recommend_data_provider<T, ShapeT, true> {
            static constexpr size_t M = decltype(std::declval<ShapeT>().magnitude())::value;
            using type = std::array<T, M>;
        };
    }

    // tensor
    template <class T, class ShapeT> using tensor =
        tensor_layout<ShapeT, typename details::_recommend_data_provider<T, ShapeT>::type>;

    // tensor_sparse
    template <class T, class ShapeT> using tensor_sparse =
        tensor_layout<ShapeT, tdp::dictionary<size_t, T>>;



    // vec_
    template <class T, size_t N> using vec_ =
        tensor<T, tensor_shape<size_t, const_size<N>>>;
    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;
    using vec4 = vec_<double, 4>;
    template <class T> using vecx_ =
        tensor<T, tensor_shape<size_t, size_t>>;
    using vecx = vecx_<double>;
    using vecsp = tensor_sparse<double, tensor_shape<size_t, size_t>>;

    // mat_
    template <class T, size_t M, size_t N> using mat_ =
        tensor<T, tensor_shape<size_t, const_size<M>, const_size<N>>>;
    using mat2x2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3x2 = mat_<double, 3, 2>;
    using mat3x3 = mat_<double, 3, 3>;
    template <class T> using matx_ =
        tensor<T, tensor_shape<size_t, size_t, size_t>>;
    using matx = matx_<double>;
    using matsp = tensor_sparse<double, tensor_shape<size_t, size_t, size_t>>;

    // cube_
    template <class T, size_t S1, size_t S2, size_t S3> using cube_ =
        tensor<T, tensor_shape<size_t, const_size<S1>, const_size<S2>, const_size<S3>>>;
    template <class T> using cubex_ =
        tensor<T, tensor_shape<size_t, size_t, size_t, size_t>>;
    using cubex = cubex_<double>;
    using cubesp = tensor_sparse<double, tensor_shape<size_t, size_t, size_t, size_t>>;



}