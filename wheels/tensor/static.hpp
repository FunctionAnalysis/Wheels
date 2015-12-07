#pragma once

#include "../core/serialize.hpp"

#include "tensor.hpp"

namespace wheels {

    template <class T, size_t ... Ss>
    class static_tensor : public serializable<static_tensor<T, Ss ...>> {
    public:
        constexpr static_tensor() {}
        template <class ... EleTs>
        constexpr static_tensor(EleTs && ... eles) 
            : _data{ {forward<EleTs>(eles) ...} } {}
        
        template <class AnotherT, class = std::enable_if_t<
            is_tensor<AnotherT, member_op_assign>::value>>
        static_tensor(const AnotherT & another) {
            assign_elements(*this, another);
        }
        template <class AnotherT, class = std::enable_if_t<
            is_tensor<AnotherT, member_op_assign>::value>>
        static_tensor & operator = (const AnotherT & another) {
            assign_elements(*this, another);
            return *this;
        }

    public:
        constexpr auto shape() const {
            return make_shape(const_size<Ss>() ...);
        }
        template <class ... SubTs>
        constexpr const T & operator()(const SubTs & ... subs) const {
            return _data[sub2ind(shape(), subs ...)];
        }
        template <class ... SubTs>
        T & operator()(const SubTs & ... subs) {
            return _data[sub2ind(shape(), subs ...)];
        }
        template <class IndexT>
        constexpr const T & operator[](const IndexT & ind) const {
            return _data[ind];
        }
        template <class IndexT>
        T & operator[](const IndexT & ind) {
            return _data[ind];
        }

    public:
        template <class V>
        decltype(auto) fields(V && visitor) {
            visitor(_data);
        }
        template <class V>
        decltype(auto) fields(V && visitor) const {
            visitor(_data);
        }

    private:
        std::array<T, prod(Ss ...)> _data;
    };

    template <class T, size_t ... Ss, class OpT>
    struct category_for_overloading<static_tensor<T, Ss...>, OpT> {
        using type = category_tensor<tensor_shape<size_t, Ss...>, T, static_tensor<T, Ss...>>;
    };


    // necessary
    template <class T, size_t ... Ss>
    constexpr auto shape_of(const static_tensor<T, Ss ...> & t) {
        return t.shape();
    }
    template <class T, size_t ... Ss, class ... SubTs>
    constexpr const T & element_at(const static_tensor<T, Ss...> & t, const SubTs & ... subs) {
        return t(subs ...);
    }
    template <class T, size_t ... Ss, class ... SubTs>
    T & element_at(static_tensor<T, Ss...> & t, const SubTs & ... subs) {
        return t(subs ...);
    }

    // auxiliary
    template <class T, size_t ... Ss, class IndexT>
    constexpr const T & element_at_index(const static_tensor<T, Ss...> & t, const IndexT & ind) {
        return t[ind];
    }
    template <class T, size_t ... Ss, class IndexT>
    T & element_at_index(static_tensor<T, Ss...> & t, const IndexT & ind) {
        return t[ind];
    }

    template <class T, size_t ... Ss, class ST, class ... SizeTs>
    void reserve_shape(static_tensor<T, Ss...> & t, const tensor_shape<ST, SizeTs...> & shape) {
        assert(t.shape() == shape);
    }

    template <class FunT, class T, size_t ... Ss, class ... Ts>
    void for_each_element(FunT && fun, const static_tensor<T, Ss...> & t, Ts && ... ts) {
        assert(all_same(shape_of(t), shape_of(ts) ...));
        for (size_t i = 0; i < numel(t); i++) {
            fun(element_at_index(t, i), element_at_index(ts, i), ...);
        }
    }
    template <class FunT, class T, size_t ... Ss, class ... Ts>
    void for_each_element(FunT && fun, static_tensor<T, Ss...> & t, Ts && ... ts) {
        assert(all_same(shape_of(t), shape_of(ts) ...));
        for (size_t i = 0; i < numel(t); i++) {
            fun(element_at_index(t, i), element_at_index(ts, i), ...);
        }
    }




}