#pragma once

#include "../core/constants.hpp"

namespace wheels {

    constexpr size_t dynamic_size = (size_t)(-1);

    template <size_t ... Sizes> class tensor_shape;

    //// tensor_shape
    //template <>
    //class tensor_shape<> {
    //public:
    //    static constexpr size_t degree = 0;
    //    static constexpr size_t static_magnitude = 1;

    //    tensor_shape() {}
    //    size_t magnitude() const { return 1; }

    //    size_t sub2ind() const { return 0u; }
    //    void ind2sub(size_t ind) const {}

    //    template <class SubsIterT>  size_t sub2ind_by_iter(SubsIterT subs_iter) const { return 0u; }
    //    template <class SubsIterT>  void ind2sub_by_iter(size_t ind, SubsIterT subs_iter) const {}

    //    template <size_t Idx> size_t at(index<Idx>) const { return 0u; }
    //    template <size_t Idx> void resize(index<Idx>, size_t) {}

    //    template <class FunT, class ... Ts>
    //    void for_each_subscript(FunT && fun, Ts && ... args) const {
    //        fun(args ...);
    //    }
    //    template <class FunT, class ... Ts>
    //    void parallel_for_each_subscript(const FunT & fun, Ts && ... args) const {
    //        fun(args ...);
    //    }
    //    template <class FunT, class ... Ts>
    //    bool for_each_subscript_if(FunT && fun, Ts && ... args) const {
    //        return fun(args ...);
    //    }
    //    template <size_t Idx, class FunT, class ... Ts>
    //    std::enable_if_t<Idx == 0> for_each_subscript_until(index<Idx>, FunT && fun, Ts && ... args) const {
    //        fun(args ...);
    //    }

    //    bool operator == (const tensor_shape &) const { return true; }
    //};

}