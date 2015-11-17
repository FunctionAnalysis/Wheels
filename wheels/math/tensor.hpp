#pragma once

#include <array>
#include <vector>

#include "../core/const_expr.hpp"
#include "../core/type.hpp"

#include "functors.hpp"
#include "tensor_shape.hpp"

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
        constexpr auto shape() const { return derived().shape_impl(); }
        template <class T, T Idx> 
        constexpr auto size(const const_ints<T, Idx> & i) const {
            return shape().at(i);
        }
        constexpr auto degree() const { return shape().degree(); }
        auto numel() const { return shape().magnitude(); }        
        
        // at_index related
        constexpr const auto & at_index(size_t index) const { return derived().at_index_impl(index); }
        template <class E> 
        constexpr const auto & operator[](const E & e) const {
            return at_index(details::_eval_const_expr(e, numel()));
        }

        // at_subs related
        template <class ... SubTs>
        constexpr const auto & at_subs(const SubTs & ... subs) const { return derived().at_subs_impl(subs ...); }
        template <class ... SubEs>
        constexpr const auto & operator()(const SubEs & ... subes) const {
            return _parenthesis_seq(std::index_sequence_for<SubEs...>(), subes ...);
        }



    private:
        template <class ... SubEs, size_t ... Is>
        constexpr const auto & _parenthesis_seq(std::index_sequence<Is...>, const SubEs & ... subes) const {
            return at_subs(details::_eval_const_expr(subes, size(const_index<Is>())) ...);
        }

    };





    template <class T, class ShapeT, class StorageT>
    class tensor_static_shaped : public tensor_base<tensor_static_shaped<T, ShapeT, StorageT>> {
    public:
        using value_type = T;
        using shape_type = ShapeT;
        using processor_type = processor::cpu;

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");
        static_assert(shape_type::is_static(), "ShapeT should be static");       

    public:
        constexpr tensor_static_shaped() {}
        constexpr tensor_static_shaped(std::initializer_list<T> ilist) {
            std::copy(ilist.begin(), ilist.end(), std::begin(_storage));
        }

        constexpr shape_type shape_impl() const { return shape_type(); }
        constexpr const value_type & at_index_impl(size_t index) const { return _storage[index]; }
        template <class ... SubTs>
        constexpr const value_type & at_subs_impl(const SubTs & ... subs) const {
            return _storage[shape_type().sub2ind(subs ...)];
        }

    private:
        StorageT _storage;
    };

    template <class T, size_t N> 
    using vec_ = tensor_static_shaped<T, tensor_shape<const_size<N>>, std::array<T, N>>;

    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;

    template <class T, size_t M, size_t N> 
    using mat_ = tensor_static_shaped<T, tensor_shape<const_size<M>, const_size<N>>, std::array<T, M * N>>;
    
    using mat2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3 = mat_<double, 3, 3>;




    namespace details {
        
    }



    template <class T, class ShapeT, class StorageT>
    class tensor_dynamic_shaped : public tensor_base<tensor_dynamic_shaped<T, ShapeT, StorageT>> {
    public:
        using value_type = T;
        using shape_type = ShapeT;
        using processor_type = processor::cpu;

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");
        static_assert(!shape_type::is_static(), "ShapeT should not be static");

    public:
        constexpr tensor_dynamic_shaped() {}
        constexpr tensor_dynamic_shaped(std::initializer_list<T> ilist)
            : _shape(ilist.size()), _storage(ilist) {
        }
        template <class SizeT, class ... SizeTs>
        constexpr tensor_dynamic_shaped(std::initializer_list<T> ilist, const SizeT & s, const SizeTs & ... ss)
            : _shape(ilist.size(), s, ss ...), _storage(ilist) {
        }

        constexpr const shape_type & shape_impl() const { return _shape; }
        constexpr const value_type & at_index_impl(size_t index) const { return _storage[index]; }
        template <class ... SubTs>
        constexpr const value_type & at_subs_impl(const SubTs & ... subs) const {
            return _storage[_shape.sub2ind(subs ...)];
        }

    private:
        ShapeT _shape;
        StorageT _storage;
    };

    template <class T> using vecx_ = tensor_dynamic_shaped<T, tensor_shape<size_t>, std::vector<T>>;
    using vecx = vecx_<double>;

    


    template <class T, class ShapeT, class StorageT>
    class tensor_gpu : public tensor_base<tensor_gpu<T, ShapeT, StorageT>> {
    public:
        using value_type = T;
        using shape_type = ShapeT;
        using processor_type = processor::gpu;

        static_assert(is_tensor_shape<shape_type>::value, "ShapeT should be a tensor_shape");
        static_assert(!shape_type::is_static(), "ShapeT should not be static");

    public:
        constexpr tensor_gpu() {}

        constexpr const shape_type & shape_impl() const { return _shape; }
        constexpr const value_type & at_index_impl(int index) const restrict(cpu, amp) { return _storage[index]; }
        template <class ... SubTs>
        constexpr const value_type & at_subs_impl(const SubTs & ... subs) const restrict(cpu, amp) {
            return _storage(subs ...);
        }

    private:
        const ShapeT _shape;
        StorageT _storage;
    };

    template <class T> using vecx_gpu_ = tensor_gpu<T, tensor_shape<size_t>, concurrency::array<T, 1>>;
    using vecx_gpu = vecx_gpu_<double>;


}