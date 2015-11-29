#pragma once

#include "../core/operators.hpp"
#include "tensor.hpp"

namespace wheels {

    // constants
    namespace ts_traits {
        // readable
        template <class ShapeT, class E>
        struct readable_at_index<ts_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class IndexT>
        constexpr const E & at_index_const_impl(const ts_category<ShapeT, constant<E>> & a, const IndexT & ind) {
            return a.data_provider().val;
        }

        template <class ShapeT, class E>
        struct readable_at_subs<ts_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class ... SubTs>
        constexpr const E & at_subs_const_impl(const ts_category<ShapeT, constant<E>> & a, const SubTs & ... subs) {
            return a.data_provider().val;
        }
    }

    template <class T, class ST, class ... SizeTs>
    constexpr auto constants(const tensor_shape<ST, SizeTs ...> & shape, T && val) {
        return compose_category(shape, constant<std::decay_t<T>>(forward<T>(val)));
    }

    // zeros
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_category(shape, constant<T>(0));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return compose_category(make_shape<ST>(sizes...), constant<T>(0));
    }

    // ones
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_category(shape, constant<T>(1));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return compose_category(make_shape<ST>(sizes...), constant<T>(1));
    }


    
    // ewise op
    template <class T, class OpT, class ... InputTs>
    struct ewise_op_result {
        using value_type = T;
        using op_type = OpT;
        OpT op;
        std::tuple<InputTs ...> inputs;
        
        template <class OpTT>
        constexpr explicit ewise_op_result(OpTT && o, InputTs && ... ins)
            : op(forward<OpTT>(o)), inputs(std::forward_as_tuple(ins) ...) {}
        
        template <class Archive> 
        void serialize(Archive & ar) { 
            ar(op, inputs); 
        }
    };

    namespace ts_traits {
       
        // readable
        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_index<ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs, size_t ... Is>
        constexpr decltype(auto) _ewise_op_result_at_index_const_impl_seq(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const IndexT & ind, const const_ints<size_t, Is...> &) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_index_const(ind) ...);
        }
        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs>
        constexpr decltype(auto) at_index_const_impl(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a, 
            const IndexT & ind) {
            return _ewise_op_result_at_index_const_impl_seq(a, ind, 
                make_const_sequence(const_size<sizeof...(InputTs)>()));
        }

        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_subs<ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class ... InputTs, size_t ... Is, class ... SubTs>
        constexpr decltype(auto) _ewise_op_result_at_subs_const_impl_seq(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const const_ints<size_t, Is...> &, const SubTs & ... subs) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_subs_const(subs ...) ...);
        }
        template <class ShapeT, class T, class OpT, class ... InputTs, class ... SubTs>
        constexpr decltype(auto) at_subs_const_impl(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const SubTs & ... subs) {
            return _ewise_op_result_at_subs_const_impl_seq(a, 
                make_const_sequence(const_size<sizeof...(InputTs)>()), subs ...);
        }
    }







}