#pragma once

#include <array>
#include <vector>

#include "../core/const_expr.hpp"
#include "../core/types.hpp"

#include "platform.hpp"
#include "tensor_shape.hpp"
#include "storage_traits.hpp"

namespace wheels {


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

    template <class DerivedT>
    class tensor_base {
    public:
        using this_t = tensor_base<DerivedT>;
        using derived_type = DerivedT;

    public:
        const derived_type & derived() const & { return (const derived_type &)(*this); }
        derived_type & derived() & { return (derived_type &)(*this); }
        derived_type && derived() && { return (derived_type &&)(*this); }

        // shape related
        constexpr decltype(auto) shape() const { return derived().shape_impl(); }
        template <class T, T Idx> 
        constexpr auto size(const const_ints<T, Idx> & i) const {
            return shape().at(i);
        }
        constexpr auto degree() const { return shape().degree(); }
        auto numel() const { return shape().magnitude(); }        
        
        // at_index related
        constexpr decltype(auto) at_index(size_t index) const { return derived().at_index_impl(index); }
        template <class E> 
        constexpr decltype(auto) operator[](const E & e) const {
            return at_index(details::_eval_const_expr(e, numel()));
        }
        decltype(auto) at_index(size_t index) { return derived().at_index_impl(index); }
        template <class E>
        decltype(auto) operator[](const E & e) {
            return at_index(details::_eval_const_expr(e, numel()));
        }

        // at_subs related
        template <class ... SubTs>
        constexpr decltype(auto) at_subs(const SubTs & ... subs) const {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return derived().at_subs_impl(subs ...); 
        }
        template <class ... SubEs>
        constexpr decltype(auto) operator()(const SubEs & ... subes) const {
            return _parenthesis_seq(std::index_sequence_for<SubEs...>(), subes ...);
        }
        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>::all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return derived().at_subs_impl(subs ...);
        }
        template <class ... SubEs>
        decltype(auto) operator()(const SubEs & ... subes) {
            return _parenthesis_seq(std::index_sequence_for<SubEs...>(), subes ...);
        }



    private:
        template <class ... SubEs, size_t ... Is>
        constexpr decltype(auto) _parenthesis_seq(std::index_sequence<Is...>, const SubEs & ... subes) const {
            return at_subs(details::_eval_const_expr(subes, size(const_index<Is>())) ...);
        }

    };


    namespace details {
        template <class Name> 
        struct _constexpr_struct {
            constexpr _constexpr_struct() {}
        };       
    }
    struct _with_shape {};
    constexpr details::_constexpr_struct<_with_shape> with_shape;
    struct _with_elements {};
    constexpr details::_constexpr_struct<_with_elements> with_elements;
    struct _with_args {};
    constexpr details::_constexpr_struct<_with_args> with_args;


    // tensor with non-static shape
    template <class ShapeT, class StorageT, class PlatformT, bool ShapeIsStatic = ShapeT::is_static()>
    class tensor : public tensor_base<tensor<ShapeT, StorageT, PlatformT, ShapeIsStatic>> {
    public:  
        friend class tensor_base<tensor<ShapeT, StorageT, PlatformT, ShapeIsStatic>>;
        using shape_type = ShapeT;
        using storage_type = StorageT;
        using platform_type = PlatformT;
        static constexpr platform_type platform() { return platform_type(); }

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");
        
        using _st_native_construct = storage_native_construct_traits<storage_type>;
        using _st_construct = storage_construct_traits<storage_type>;
        using value_type = typename storage_type::value_type;
        
    public:
        template <wheels_enable_if(_st_native_construct::is_default_constructible)>
        constexpr tensor() {}

        template <wheels_enable_if(_st_native_construct::is_copy_constructible)>
        constexpr explicit tensor(const ShapeT & s, const StorageT & stg)
            : _shape(s), _storage(stg) {}
        
        template <wheels_enable_if(_st_construct::is_constructible_with_shape)>
        constexpr explicit tensor(const ShapeT & s)
            : _shape(s), _storage(_st_construct::construct_with_shape(s)) {}

        template <wheels_enable_if(_st_construct::is_constructible_with_shape), class ... SizeTs>
        constexpr explicit tensor(details::_constexpr_struct<_with_shape>, const SizeTs & ... ss)
            : _shape(ss ...), _storage(_st_construct::construct_with_shape(_shape)) {}

        template <wheels_enable_if(_st_construct::is_constructible_with_shape_elements), class ... EleTs>
        constexpr tensor(const ShapeT & s, details::_constexpr_struct<_with_elements>, EleTs && ... eles)
            : _shape(s), _storage(_st_construct::construct_with_shape_elements(_shape, forward<EleTs>(eles) ...)) {}

        template <wheels_enable_if(_st_construct::is_constructible_with_shape_args), class ... ArgTs>
        constexpr tensor(const ShapeT & s, details::_constexpr_struct<_with_args>, ArgTs && ... args)
            : _shape(s), _storage(_st_construct::construct_with_shape_args(_shape, forward<ArgTs>(args) ...)) {}

        constexpr const StorageT & storage() const { return _storage; }
        constexpr const ShapeT & shape_impl() const {
            return _shape;
        }

    private:
        StorageT & storage() { return _storage; }     
    private:
        ShapeT _shape;
        StorageT _storage;
    };


    // tensor with static shape
    template <class ShapeT, class StorageT, class PlatformT>
    class tensor<ShapeT, StorageT, PlatformT, true> : public tensor_base<tensor<ShapeT, StorageT, PlatformT, true>> {
    public:
        friend class tensor_base<tensor<ShapeT, StorageT, PlatformT, true>>;
        using shape_type = ShapeT;
        using storage_type = StorageT;
        using platform_type = PlatformT;
        static constexpr platform_type platform() { return platform_type(); }

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");

        using _st_native_construct = storage_native_construct_traits<storage_type>;
        using _st_construct = storage_construct_traits<storage_type>;
        using value_type = typename storage_type::value_type;

    public:
        template <wheels_enable_if(_st_construct::is_constructible_with_shape)>
        constexpr tensor() : _storage(_st_construct::construct_with_shape(ShapeT())) {}

        template <wheels_enable_if(_st_construct::is_constructible_with_shape_elements), class ... EleTs>
        constexpr tensor(const details::_constexpr_struct<_with_elements> &, EleTs && ... eles)
            : _storage(_st_construct::construct_with_shape_elements(ShapeT(), forward<EleTs>(eles) ...)) {}

        template <wheels_enable_if(_st_construct::is_constructible_with_shape_args), class ... ArgTs>
        constexpr tensor(details::_constexpr_struct<_with_args>, ArgTs && ... args)
            : _storage(_st_construct::construct_with_shape_args(ShapeT(), forward<ArgTs>(args) ...)) {}

        constexpr const StorageT & storage() const { return _storage; }
        constexpr ShapeT shape_impl() const {
            return ShapeT();
        }

    private:
        StorageT & storage() { return _storage; }
    private:
        StorageT _storage;
    };






}