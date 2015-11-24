#pragma once

#include <cassert>
#include <array>
#include <vector>

#include "../core/const_expr.hpp"
#include "../core/types.hpp"

#include "tensor_shape.hpp"
#include "tensor_data.hpp"
#include "tensor_assign.hpp"

namespace wheels {

    // forward declaration
    template <class LayoutT>
    class tensor_base;

    template <class ShapeT, class DataProviderT,
        bool ShapeIsStatic = ShapeT::is_static>
    class tensor_layout;

    namespace details {
        template <class ShapeT>
        struct _shape_property {};
        template <class ShapeT, class DataProviderT, bool S>
        struct _shape_property<tensor_layout<ShapeT, DataProviderT, S>> {
            static constexpr bool is_static = ShapeT::is_static;
            static constexpr bool is_scalar = ShapeT::degree == 0;
            static constexpr bool is_vector = ShapeT::degree == 1;
            static constexpr bool is_matrix = ShapeT::degree == 2;
            static constexpr bool is_cube = ShapeT::degree == 3;
        };

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

    template <class LayoutT,
        bool EleReadableAtIndex = details::_layout_property<LayoutT>::element_readable_at_index,
        bool EleReadableAtSubs = details::_layout_property<LayoutT>::element_readable_at_subs>
    class tensor_method_read_element;

    template <class LayoutT,
        bool EleWritableAtIndex = details::_layout_property<LayoutT>::element_writable_at_index,
        bool EleWritableAtSubs = details::_layout_property<LayoutT>::element_writable_at_subs>
    class tensor_method_write_element;

    template <class LayoutT,
        bool AssignByIndex = details::_layout_property<LayoutT>::element_readable_at_index>
    class tensor_method_assign;

    template <class LayoutT>
    class tensor_all_methods;



    // tensor_base
    template <class LayoutT>
    class tensor_base {
    public:
        using layout_type = LayoutT;

        constexpr const layout_type & layout() const { return (const layout_type &)(*this); }
        layout_type & layout() { return (layout_type &)(*this); }

        // shape related
        constexpr decltype(auto) shape() const { return layout().shape_impl(); }
        constexpr auto degree() const { return const_ints<int, decltype(shape())::degree>(); }
        constexpr auto degree_sequence() const { return make_const_sequence(degree()); }

        template <class T, T Idx>
        constexpr auto size(const const_ints<T, Idx> & i) const {
            return shape().at(i);
        }
        auto numel() const { return shape().magnitude(); }

        // storage
        constexpr decltype(auto) storage() const { return layout().data_provider_impl(); }     
    };


#define WHEELS_TENSOR_METHOD_LAYER(method_name) \
    constexpr const tensor_method_##method_name<LayoutT> & \
        method_##method_name() const { \
        return (const tensor_method_##method_name<LayoutT> &)(*this); \
    } \
    tensor_method_##method_name<LayoutT> & \
        method_##method_name() { \
        return (tensor_method_##method_name<LayoutT> &)(*this); \
    }



    // tensor_method_read_element
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, true, true>
        : public tensor_base<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(read_element)
        static constexpr bool element_readable_at_index = true;
        static constexpr bool element_readable_at_subs = true;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index(const IndexT & index) const {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, false, true>
        : public tensor_base<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(read_element)
        static constexpr bool element_readable_at_index = false;
        static constexpr bool element_readable_at_subs = true;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index(const IndexT & index) const {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
            });
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, true, false>
        : public tensor_base<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(read_element)
        static constexpr bool element_readable_at_index = true;
        static constexpr bool element_readable_at_subs = false;
        // read element at index
        template <class IndexT>
        constexpr decltype(auto) at_index(const IndexT & index) const {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // read element at subs
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_index(layout().data_provider_impl(),
                sub2ind(shape(), subs ...));
        }
    };
    template <class LayoutT>
    class tensor_method_read_element<LayoutT, false, false>
        : public tensor_base<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(read_element)
        static constexpr bool element_readable_at_index = false;
        static constexpr bool element_readable_at_subs = false;
    };


    

    // tensor_method_write_element
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, true, true>
        : public tensor_method_read_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(write_element)
        static constexpr bool element_writable_at_index = true;
        static constexpr bool element_writable_at_subs = true;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index(const IndexT & index) {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, false, true>
        : public tensor_method_read_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(write_element)
        static constexpr bool element_writable_at_index = false;
        static constexpr bool element_writable_at_subs = true;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index(const IndexT & index) {
            return invoke_with_subs(shape(), index, [this](const auto & subs ...) {
                return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
            });
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_subs(layout().data_provider_impl(), subs ...);
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, true, false>
        : public tensor_method_read_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(write_element)
        static constexpr bool element_writable_at_index = true;
        static constexpr bool element_writable_at_subs = false;
        // write element at index
        template <class IndexT>
        decltype(auto) at_index(const IndexT & index) {
            return tdp::element_at_index(layout().data_provider_impl(), index);
        }
        // write element at subs
        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return tdp::element_at_index(layout().data_provider_impl(),
                sub2ind(shape(), subs ...));
        }
    };
    template <class LayoutT>
    class tensor_method_write_element<LayoutT, false, false>
        : public tensor_method_read_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(write_element)
        static constexpr bool element_writable_at_index = false;
        static constexpr bool element_writable_at_subs = false;
    };








    // tensor_method_assign
    template <class LayoutT>
    class tensor_method_assign<LayoutT, true> // by index
        : public tensor_method_write_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(assign)
        static constexpr bool assign_by_index = true;
        // copy_to
        template <class LayoutToT, bool B1, bool B2>
        void copy_to(tensor_method_write_element<LayoutToT, B1, B2> & to) const {
            static_assert(B1 || B2, "'to' is not writable");
            assert(shape() == to.shape());
            tdp::reserve_storage(to.shape(), to.layout().data_provider_impl());
            for (int ind = 0; ind < numel(); ind++) {
                to.at_index(ind) = method_read_element().at_index(ind);
            }
        }
    };
    template <class LayoutT>
    class tensor_method_assign<LayoutT, false> // by subs
        : public tensor_method_write_element<LayoutT> {
    public:
        WHEELS_TENSOR_METHOD_LAYER(assign)
        static constexpr bool assign_by_index = false;
        // copy_to
        template <class LayoutToT, bool B1, bool B2>
        void copy_to(tensor_method_write_element<LayoutToT, B1, B2> & to) const {
            static_assert(B1 || B2, "'to' is not writable");
            assert(shape() == to.shape());
            tdp::reserve_storage(to.shape(), to.layout().data_provider_impl());
            for_each_subscript(shape(), [this, &to](const auto & ... subs) {
                to.at_subs(subs ...) = method_read_element().at_subs(subs ...);
            });
        }
    };




    namespace details{
        template <class LayoutT> 
        using _tensor_method_last = tensor_method_assign<LayoutT>;
    }



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

    // tensor_all_methods
    template <class LayoutT>
    class tensor_all_methods 
        : public details::_tensor_method_last<LayoutT> {
    public:
        using layout_type = LayoutT;

        // [...] based on at_index
        template <class E>
        constexpr decltype(auto) operator[](const E & e) const {
            return at_index(details::_eval_const_expr(e, numel()));
        }
        template <class E>
        decltype(auto) operator[](const E & e) {
            return at_index(details::_eval_const_expr(e, numel()));
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
            return method_read_element().
                at_subs(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
        }
        template <class ... SubEs, int ... Is>
        decltype(auto) _parenthesis_seq(const_ints<int, Is...>, const SubEs & ... subes) {
            return method_write_element().
                at_subs(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
        }
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
        : public tensor_all_methods<tensor_layout<ShapeT, DataProviderT, false>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = false;

        template <class ST, class DPT>
        friend constexpr tensor_layout<ST, std::decay_t<DPT>>
            compose_tensor_layout(const ST & shape, DPT && dp);

    public:  
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT; 
        
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
        template <class LayoutFromT, bool B>
        tensor_layout(const tensor_method_assign<LayoutFromT, B> & from) : _shape(from.shape()) {
            from.copy_to(*this);
        }
        template <class LayoutFromT, bool B>
        tensor_layout & operator = (const tensor_method_assign<LayoutFromT, B> & from) {
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
        : public tensor_all_methods<tensor_layout<ShapeT, DataProviderT, true>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = true;

        template <class ST, class DPT>
        friend constexpr tensor_layout<ST, std::decay_t<DPT>>
            compose_tensor_layout(const ST & shape, DPT && dp);

    public:
        using shape_type = ShapeT;
        using data_provider_type = DataProviderT;

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
        template <class LayoutFromT, bool B>
        tensor_layout(const tensor_method_assign<LayoutFromT, B> & from) {
            from.copy_to(*this);
        }
        template <class LayoutFromT, bool B>
        tensor_layout & operator = (const tensor_method_assign<LayoutFromT, B> & from) {
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

    // is_tensor_layout
    template <class T> struct is_tensor_layout : no {};
    template <class ShapeT, class DataProviderT, bool B>
    struct is_tensor_layout<tensor_layout<ShapeT, DataProviderT, B>> : yes {};


    template <class ST, class DPT>
    constexpr tensor_layout<ST, std::decay_t<DPT>> compose_tensor_layout(const ST & shape, DPT && dp) {
        return tensor_layout<ST, std::decay_t<DPT>>(shape, forward<DPT>(dp));
    }


    template <class T, size_t N> using vec_ = 
        tensor_layout<tensor_shape<size_t, const_size<N>>, std::array<T, N>>;
    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;
    using vec4 = vec_<double, 4>;
    template <class T> using vecx_ = 
        tensor_layout<tensor_shape<size_t>, std::vector<T>>;
    using vecx = vecx_<double>;

    template <class T, size_t M, size_t N> using mat_ =
        tensor_layout<tensor_shape<size_t, const_size<M>, const_size<N>>, std::array<T, M * N>>;
    using mat2x2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3x2 = mat_<double, 3, 2>;
    using mat3x3 = mat_<double, 3, 3>;
    template <class T> using matx_ = 
        tensor_layout<tensor_shape<size_t, size_t, size_t>, std::vector<T>>;
    using matx = matx_<double>;

    template <class T, size_t S1, size_t S2, size_t S3> using cube_ =
        tensor_layout<tensor_shape<size_t, const_size<S1>, const_size<S2>, const_size<S3>>, std::array<T, S1 * S2 * S3>>;
    template <class T> using cubex_ =
        tensor_layout<tensor_shape<size_t, size_t, size_t, size_t>, std::vector<T>>;
    using cubex = cubex_<double>;
}