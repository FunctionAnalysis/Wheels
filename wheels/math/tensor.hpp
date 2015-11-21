#pragma once

#include <array>
#include <vector>

#include "../core/const_expr.hpp"
#include "../core/types.hpp"
#include "../core/platforms.hpp"

#include "tensor_shape.hpp"
#include "tensor_data.hpp"

namespace wheels {

    // forward declaration
    template <class PlatformT, class LayoutT>
    class tensor_base;
    
    template <class ShapeT, class StorageT, class PlatformT,
        bool ShapeIsStatic = ShapeT::is_static>
    class tensor_layout;

    namespace details {
        template <class LayoutT>
        struct _ele_accessible_at_index : no {};
        template <class ShapeT, class StorageT, class PlatformT, bool S>
        struct _ele_accessible_at_index<tensor_layout<ShapeT, StorageT, PlatformT, S>> 
            : const_bool<is_element_accessible_at_index<PlatformT, StorageT>::value> {};
    }
    template <class PlatformT, class LayoutT,
        bool SupportEleAtIndex = details::_ele_accessible_at_index<LayoutT>::value>
    class tensor_manip_element_at_index;

    namespace details {
        template <class LayoutT>
        struct _ele_accessible_at_subs : no {};
        template <class ShapeT, class StorageT, class PlatformT, bool S>
        struct _ele_accessible_at_subs<tensor_layout<ShapeT, StorageT, PlatformT, S>>
            : const_bool<is_element_accessible_at_subs<PlatformT, StorageT>::value> {};
    }
    template <class PlatformT, class LayoutT,
        bool SupportEleAtSubs = details::_ele_accessible_at_subs<LayoutT>::value>
    class tensor_manip_element_at_subs;

    template <class PlatformT, class LayoutT>
    class tensor_all_methods;




    // tensor_base
    template <class LayoutT>
    class tensor_base<platform_cpu, LayoutT> {
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        const layout_type & layout() const { return (const layout_type &)(*this); }
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
        constexpr decltype(auto) storage() const { return layout().storage_impl(); }
    };






    // tensor_manip_element_at_index
    template <class LayoutT>
    class tensor_manip_element_at_index<platform_cpu, LayoutT, true> 
        : public tensor_base<platform_cpu, LayoutT> {        
        static constexpr bool _is_element_accessible_at_index = true;
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        // at_index related
        template <class IndexT>
        constexpr decltype(auto) at_index(const IndexT & index) const {
            return element_at_index(layout().storage_impl(), index);
        }

        template <class IndexT>
        decltype(auto) at_index(const IndexT & index) {
            return element_at_index(layout().storage_impl(), index);
        }
    };

    template <class LayoutT>
    class tensor_manip_element_at_index<platform_cpu, LayoutT, false>
        : public tensor_base<platform_cpu, LayoutT> {
        static constexpr bool _is_element_accessible_at_index = false;
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        // at_index related
        template <class IndexT>
        constexpr decltype(auto) at_index(const IndexT & index) const {
            return _at_subs_seq(index, degree_sequence());
        }

        template <class IndexT>
        decltype(auto) at_index(const IndexT & index) {
            return _at_subs_seq(index, degree_sequence());
        }

    private:
        template <class IndexT, int ... Is>
        constexpr decltype(auto) _at_subs_seq(const IndexT & index, const_ints<int, Is...>) const {
            size_t subs[sizeof...(Is)];
            ind2sub(shape(), index, subs[Is]...);
            return element_at_subs(layout().storage_impl(), subs[Is]...);
        }

        template <class IndexT, int ... Is>
        decltype(auto) _at_subs_seq(const IndexT & index, const_ints<int, Is...>) {
            size_t subs[sizeof...(Is)];
            ind2sub(shape(), index, subs[Is]...);
            return element_at_subs(layout().storage_impl(), subs[Is]...);
        }
    };







    
    template <class LayoutT>
    class tensor_manip_element_at_subs<platform_cpu, LayoutT, true> 
        : public tensor_manip_element_at_index<platform_cpu, LayoutT> {
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        // at_subs related
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return element_at_subs(layout().storage_impl(), subs ...);
        }

        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return element_at_subs(layout().storage_impl(), subs ...);
        }      
    };

    template <class LayoutT>
    class tensor_manip_element_at_subs<platform_cpu, LayoutT, false>
        : public tensor_manip_element_at_index<platform_cpu, LayoutT> {
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        // at_subs related
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return element_at_index(layout().storage_impl(), sub2ind(shape(), subs ...));
        }

        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return element_at_index(layout().storage_impl(), sub2ind(shape(), subs ...));
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


    template <class LayoutT>
    class tensor_all_methods<platform_cpu, LayoutT> 
        : public tensor_manip_element_at_subs<platform_cpu, LayoutT> {
    public:
        using platform_type = platform_cpu;
        using layout_type = LayoutT;

        // at_index
        template <class E>
        constexpr decltype(auto) operator[](const E & e) const {
            return at_index(details::_eval_const_expr(e, numel()));
        }
        template <class E>
        decltype(auto) operator[](const E & e) {
            return at_index(details::_eval_const_expr(e, numel()));
        }

        // at_subs
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
            return at_subs(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
        }
        template <class ... SubEs, int ... Is>
        decltype(auto) _parenthesis_seq(const_ints<int, Is...>, const SubEs & ... subes) {
            return at_subs(details::_eval_const_expr(subes, size(const_int<Is>())) ...);
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
    template <class ShapeT, class StorageT, class PlatformT>
    class tensor_layout<ShapeT, StorageT, PlatformT, false> 
        : public tensor_all_methods<PlatformT, tensor_layout<ShapeT, StorageT, PlatformT, false>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = false;

    public:  
        using shape_type = ShapeT;
        using storage_type = StorageT;
        using value_type = typename storage_type::value_type;
        using platform_type = PlatformT;

        static constexpr platform_type platform() { return platform_type(); }        
        
    public:
        template <wheels_enable_if(is_default_constructible<storage_type>::value)>
        constexpr tensor_layout() {}

        template <wheels_enable_if(is_copy_constructible<storage_type>::value)>
        constexpr tensor_layout(const ShapeT & s, const StorageT & stg)
            : _shape(s), _storage(stg) {}

        template <wheels_enable_if(is_move_constructible<storage_type>::value)>
        constexpr tensor_layout(const ShapeT & s, StorageT && stg)
            : _shape(s), _storage(std::move(stg)) {}

        // tensor_layout(shape)
        template <wheels_enable_if(is_constructible_with_shape<storage_type>::value)>
        constexpr explicit tensor_layout(const ShapeT & s)
            : _shape(s), _storage(construct_with_shape(types<storage_type>(), s)) {}

        // tensor_layout(shape, with_elements, elements ...)
        template <wheels_enable_if(is_constructible_with_shape_elements<storage_type>::value), 
            class ... EleTs>
        constexpr tensor_layout(const ShapeT & s, details::_tensor_construct_tag<_with_elements>, EleTs && ... eles)
            : _shape(s), _storage(construct_with_shape_elements(types<storage_type>(), _shape, forward<EleTs>(eles) ...)) {}

        // tensor_layout(shape, with_args, args ...)
        template <class ... ArgTs>
        constexpr tensor_layout(const ShapeT & s, details::_tensor_construct_tag<_with_args>, ArgTs && ... args)
            : _shape(s), _storage(construct_with_args(types<storage_type>(), forward<ArgTs>(args) ...)) {}

        // interfaces
        constexpr const ShapeT & shape_impl() const { return _shape; }
        constexpr const StorageT & storage_impl() const { return _storage; }
        StorageT & storage_impl() { return _storage; }

    private:
        ShapeT _shape;
        StorageT _storage;
    };


    // tensor_layout with static shape
    template <class ShapeT, class StorageT, class PlatformT>
    class tensor_layout<ShapeT, StorageT, PlatformT, true> 
        : public tensor_all_methods<PlatformT, tensor_layout<ShapeT, StorageT, PlatformT, true>> {
        static_assert(is_tensor_shape<ShapeT>::value, "ShapeT should be a tensor_shape");
        static constexpr bool _shape_is_static = true;

    public:
        using shape_type = ShapeT;
        using storage_type = StorageT;
        using value_type = typename storage_type::value_type;
        using platform_type = PlatformT;

        static constexpr platform_type platform() { return platform_type(); }

    public:
        template <wheels_enable_if(is_constructible_with_shape<storage_type>::value)>
        constexpr tensor_layout() : _storage(construct_with_shape(types<storage_type>(), ShapeT())) {}

        // tensor_layout(with_elements, elements ...)
        template <wheels_enable_if(is_constructible_with_shape_elements<storage_type>::value), 
            class ... EleTs>
        constexpr tensor_layout(const details::_tensor_construct_tag<_with_elements> &, EleTs && ... eles)
            : _storage(construct_with_shape_elements(types<storage_type>(), ShapeT(), forward<EleTs>(eles) ...)) {}

        template <class ... ArgTs>
        constexpr tensor_layout(details::_tensor_construct_tag<_with_args>, ArgTs && ... args)
            : _storage(construct_with_args(types<storage_type>(), forward<ArgTs>(args) ...)) {}

        // interfaces
        constexpr ShapeT shape_impl() const { return ShapeT(); }
        constexpr const StorageT & storage_impl() const { return _storage; }
        StorageT & storage_impl() { return _storage; }

    private:
        StorageT _storage;
    };



}