#pragma once

#include <array>
#include <vector>

#include "../core/const_expr.hpp"
#include "../core/types.hpp"

#include "functors.hpp"
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
            static_assert(const_ints<bool, is_int<SubTs>::value ...>().all(),
                "at_subs(...) requires all subs should be integral or const_ints");
            return derived().at_subs_impl(subs ...); 
        }
        template <class ... SubEs>
        constexpr decltype(auto) operator()(const SubEs & ... subes) const {
            return _parenthesis_seq(std::index_sequence_for<SubEs...>(), subes ...);
        }
        template <class ... SubTs>
        decltype(auto) at_subs(const SubTs & ... subs) {
            static_assert(const_ints<bool, is_int<SubTs>::value ...>().all(),
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




    template <class ShapeT, class StorageT, bool ShapeIsStatic = ShapeT::is_static()>
    class tensor : public tensor_base<tensor<ShapeT, StorageT, ShapeIsStatic>> {
    public:        
        using shape_type = ShapeT;
        using storage_type = StorageT;

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");
        
        using _stt = storage_traits<storage_type>;
        using value_type = typename _stt::value_type;
        
    public:
        template <wheels_enable_if(_stt::is_default_constructible())>
        constexpr explicit tensor() {}
        

        constexpr const StorageT & storage() const { return _storage; }
        constexpr const ShapeT & shape_impl() const {
            return _shape;
        }
        
        constexpr decltype(auto) at_index_impl(size_t index) const {
            return _stt::element(_storage, index);
        }
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_impl(const SubTs & ... subs) const {
            return _stt::element(_storage, _shape.sub2ind(subs ...));
        }

        decltype(auto) at_index_impl(size_t index) {
            return _stt::element(_storage, index);
        }
        template <class ... SubTs>
        decltype(auto) at_subs_impl(const SubTs & ... subs) {
            return _stt::element(_storage, _shape.sub2ind(subs ...));
        }

    private:
        ShapeT _shape;
        StorageT _storage;
    };



    template <class ShapeT, class StorageT>
    class tensor<ShapeT, StorageT, true> : public tensor_base<tensor<ShapeT, StorageT, true>> {
    public:
        using shape_type = ShapeT;
        using storage_type = StorageT;

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");

        using _stt = storage_traits<storage_type>;
        using value_type = typename _stt::value_type;

    public:
        template <class ... ArgTs>
        constexpr tensor(ArgTs && ... args) 
            : _storage(_stt::construct_with_size(ShapeT().magnitude(), 
                std::forward<ArgTs>(args) ...)) {
        }

        constexpr const StorageT & storage() const { return _storage; }
        constexpr ShapeT shape_impl() const {
            return ShapeT();
        }

        constexpr decltype(auto) at_index_impl(size_t index) const {
            return _storage[index];
        }
        template <class ... SubTs>
        constexpr decltype(auto) at_subs_impl(const SubTs & ... subs) const {
            return _storage[ShapeT().sub2ind(subs ...)];
        }

        decltype(auto) at_index_impl(size_t index) {
            return _storage[index];
        }
        template <class ... SubTs>
        decltype(auto) at_subs_impl(const SubTs & ... subs) {
            return _storage[ShapeT().sub2ind(subs ...)];
        }

    private:
        StorageT _storage;
    };


}