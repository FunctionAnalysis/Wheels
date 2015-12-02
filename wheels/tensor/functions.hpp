#pragma once

#include <iostream>

#include "../core/overloads.hpp"

#include "categories.hpp"

namespace wheels {


    // constants
    template <class T>
    struct constant {
        using value_type = T;

        template <class TT>
        constexpr constant(TT && v) : val(forward<TT>(v)) {}

        template <class ... ArgTs>
        constexpr const T & operator()(ArgTs && ...) const { return val; }
        
        template <class Archive> 
        void serialize(Archive & ar) { ar(val); }
        
        T val;
    };

    namespace tensor_traits {
        // readable
        template <class ShapeT, class E>
        struct readable_at_index<tensor_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class IndexT>
        constexpr const E & at_index_const_impl(const tensor_category<ShapeT, constant<E>> & a, const IndexT & ind) {
            return a.data_provider().val;
        }

        template <class ShapeT, class E>
        struct readable_at_subs<tensor_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class ... SubTs>
        constexpr const E & at_subs_const_impl(const tensor_category<ShapeT, constant<E>> & a, const SubTs & ... subs) {
            return a.data_provider().val;
        }
    }

    template <class T, class ST, class ... SizeTs>
    constexpr auto constants(const tensor_shape<ST, SizeTs ...> & shape, T && val) {
        return make_tensor(shape, constant<std::decay_t<T>>(forward<T>(val)));
    }

    // zeros
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs ...> & shape) {
        return make_tensor(shape, constant<T>(0));
    }
    template <class T = double, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return make_tensor(make_shape(sizes...), constant<T>(types<T>::zero()));
    }

    // ones
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs ...> & shape) {
        return make_tensor(shape, constant<T>(1));
    }
    template <class T = double, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return make_tensor(make_shape(sizes...), constant<T>(1));
    }



    // meshgrid
    template <class T, size_t Idx>
    struct meshgrid_result {
        using value_type = T;

        constexpr meshgrid_result(){}

        template <class Archive>
        void serialize(Archive & ar) {}
    };

    namespace tensor_traits {
        template <class ShapeT, class T, size_t Idx>
        struct readable_at_subs<tensor_category<ShapeT, meshgrid_result<T, Idx>>> : yes {};

        template <class ShapeT, class T, size_t Idx, class ... SubTs>
        constexpr T at_subs_const_impl(
            const tensor_category<ShapeT, meshgrid_result<T, Idx>> & a,
            const SubTs & ... subs) {
            return (T)std::get<Idx>(std::forward_as_tuple(subs ...));
        }
    }

    namespace details {
        template <class T, class ShapeT, size_t ... Is>
        constexpr auto _meshgrid_seq(const ShapeT & shape, const_ints<size_t, Is...>) {
            return std::make_tuple(make_tensor(shape, meshgrid_result<T, Is>())...);
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto meshgrid(const tensor_shape<ST, SizeTs ...> & shape) {
        return details::_meshgrid_seq<T>(shape, make_const_sequence(const_size<sizeof...(SizeTs)>()));
    }
    template <class T = double, class ... SizeTs>
    constexpr auto meshgrid(const SizeTs & ... sizes) {
        return details::_meshgrid_seq<T>(make_shape(sizes ...), make_const_sequence(const_size<sizeof...(SizeTs)>()));
    }




    // iota
    template <class T>
    struct iota_result {
        using value_type = T;
        constexpr iota_result(){}
        template <class Archive>
        void serialize(Archive & ar) {}
    };

    namespace tensor_traits {
        template <class ShapeT, class T>
        struct readable_at_index<tensor_category<ShapeT, iota_result<T>>> : yes {};

        template <class ShapeT, class T, class IndexT>
        constexpr T at_index_const_impl(const tensor_category<ShapeT, iota_result<T>> &, const IndexT & index) {
            return (T)index;
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto iota(const tensor_shape<ST, SizeTs ...> & shape) {
        return make_tensor(shape, iota_result<T>());
    }
    template <class T = double, class ... SizeTs>
    constexpr auto iota(const SizeTs & ... sizes) {
        return make_tensor(make_shape(sizes...), iota_result<T>());
    }





    // unit axis vector
    template <class T, size_t Idx>
    struct unit_axis_result {
        using value_type = T;
        constexpr unit_axis_result() {}
        template <class Archiver>
        void serialize(Archiver &) {}
    };

    namespace tensor_traits {
        template <class ShapeT, class T, size_t Idx>
        struct readable_at_index<tensor_category<ShapeT, unit_axis_result<T, Idx>>> : yes {};

        template <class ShapeT, class T, size_t Idx, class IndexT>
        constexpr T at_index_const_impl(const tensor_category<ShapeT, unit_axis_result<T, Idx>> & a, const IndexT & index) {
            return index == Idx ? 1 : 0;
        }
    }

    // unit_x/y/z
    template <class T = double, class SizeT = const_ints<size_t, 3>>
    constexpr auto unit_x(const SizeT & s = SizeT()) {
        return make_tensor(make_shape(s), unit_axis_result<T, 0>());
    }
    template <class T = double, class SizeT = const_ints<size_t, 3>>
    constexpr auto unit_y(const SizeT & s = SizeT()) {
        return make_tensor(make_shape(s), unit_axis_result<T, 1>());
    }
    template <class T = double, class SizeT = const_ints<size_t, 3>>
    constexpr auto unit_z(const SizeT & s = SizeT()) {
        return make_tensor(make_shape(s), unit_axis_result<T, 2>());
    }




    // eye
    template <class T>
    struct eye_result {
        using value_type = T;
        
        constexpr eye_result() {}
        
        template <class Archiver> 
        void serialize(Archiver &) {}
    };

    template <class ShapeT, class T>
    struct eye_result_nonzero_iterator 
        : indexed_iterator_base<T, size_t, eye_result_nonzero_iterator<ShapeT, T>> {       
      
        constexpr eye_result_nonzero_iterator(size_t ind, const ShapeT & s)
            : indexed_iterator_base<T, size_t, eye_result_nonzero_iterator>(ind, min_shape_size(s)),
            val(1), shape(s) {}
        constexpr const T & operator * () const { return val; }
        constexpr const T * operator -> () const { return &val; }

        const T & val;
        const ShapeT & shape;
    };

    namespace tensor_traits {
        template <class ShapeT, class T>
        struct readable_at_subs<tensor_category<ShapeT, eye_result<T>>> : yes {};

        template <class ShapeT, class T, class ... SubTs>
        constexpr T at_subs_const_impl(
            const tensor_category<ShapeT, eye_result<T>> & a,
            const SubTs & ... subs) {
            return all_same(subs ...) ? 1.0 : types<T>::zero();
        }

        template <class ShapeT, class T>
        struct nonzero_iterator_type<tensor_category<ShapeT, eye_result<T>>> {
            using type = eye_result_nonzero_iterator<ShapeT, T>;
        };
        template <class ShapeT, class T, size_t ... Is>
        constexpr size_t _iter2ind_seq(const eye_result_nonzero_iterator<ShapeT, T> & iter, const_ints<size_t, Is...>) {
            return sub2ind(iter.shape, constant<size_t>(iter.ind)(Is) ...);
        }
        template <class ShapeT, class T>
        constexpr size_t iter2ind(const eye_result_nonzero_iterator<ShapeT, T> & iter) {
            return _iter2ind_seq(iter, make_rank_sequence(iter.shape));
        }

        template <class ShapeT, class T>
        constexpr auto nzbegin_impl(const tensor_category<ShapeT, eye_result<T>> & t) {
            return eye_result_nonzero_iterator<ShapeT, T>(0, t.shape());
        }
        template <class ShapeT, class T>
        constexpr auto nzend_impl(const tensor_category<ShapeT, eye_result<T>> & t) {
            return eye_result_nonzero_iterator<ShapeT, T>(min_shape_size(t.shape()), t.shape());
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto eye(const tensor_shape<ST, SizeTs ...> & shape) {
        return make_tensor(shape, eye_result<T>());
    }
    template <class T = double, class ... SizeTs>
    constexpr auto eye(const SizeTs & ... sizes) {
        return make_tensor(make_shape(sizes ...), eye_result<T>());
    }





    
    // ewise op
    template <class T, class OpT, class ... InputTs>
    struct ewise_op_result {
        using value_type = T;
        using op_type = OpT;
        
        template <class OpTT>
        constexpr explicit ewise_op_result(OpTT && o, InputTs && ... ins)
            : op(forward<OpTT>(o)), inputs(std::forward<InputTs>(ins) ...) {}
        
        template <class Archive> 
        void serialize(Archive & ar) { 
            ar(op, inputs); 
        }

        OpT op;
        std::tuple<InputTs ...> inputs;
    };

    template <class OpT, class ... InputTs>
    constexpr auto make_tensor_ewise_op_result(OpT && op, InputTs && ... inputs) {
        using result_t = std::decay_t<decltype(op(inputs.at_index_const(0) ...))>;
        return ewise_op_result<result_t, std::decay_t<OpT>, InputTs...>(forward<OpT>(op), 
            forward<InputTs>(inputs) ...);
    }

    namespace tensor_traits {
       
        // readable
        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_index<tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs, size_t ... Is>
        constexpr decltype(auto) _ewise_op_result_at_index_const_impl_seq(
            const tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const IndexT & ind, const const_ints<size_t, Is...> &) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_index_const(ind) ...);
        }
        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs>
        constexpr decltype(auto) at_index_const_impl(
            const tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a, 
            const IndexT & ind) {
            return _ewise_op_result_at_index_const_impl_seq(a, ind, 
                make_const_sequence(const_size<sizeof...(InputTs)>()));
        }

        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_subs<tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class ... InputTs, size_t ... Is, class ... SubTs>
        constexpr decltype(auto) _ewise_op_result_at_subs_const_impl_seq(
            const tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const const_ints<size_t, Is...> &, const SubTs & ... subs) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_subs_const(subs ...) ...);
        }
        template <class ShapeT, class T, class OpT, class ... InputTs, class ... SubTs>
        constexpr decltype(auto) at_subs_const_impl(
            const tensor_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const SubTs & ... subs) {
            return _ewise_op_result_at_subs_const_impl_seq(a, 
                make_const_sequence(const_size<sizeof...(InputTs)>()), subs ...);
        }
    }



    // overloads operators concerning tensor_category
    template <class ShapeT, class DataProviderT>
    struct join_overloading<tensor_category<ShapeT, DataProviderT>> : yes {};

    // unary
    // tensor -> tensor
    template <class OpT, class ShapeT, class DataProviderT>
    struct overloaded<OpT, tensor_category<ShapeT, DataProviderT>> {
        constexpr overloaded() {}
        template <class A>
        constexpr auto operator()(A && a) const {
            return make_tensor(a.shape(), make_tensor_ewise_op_result(OpT(), forward<A>(a)));
        }
    };
    
    // binary
    // tensor + tensor -> tensor
    template <class OpT, class ShapeT1, class DataProviderT1, class ShapeT2, class DataProviderT2>
    struct overloaded<OpT, tensor_category<ShapeT1, DataProviderT1>, tensor_category<ShapeT2, DataProviderT2>> {
        constexpr overloaded() {}
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(a.shape() == b.shape());
            return make_tensor(a.shape(), make_tensor_ewise_op_result(OpT(), forward<A>(a), forward<B>(b)));
        }
    };
    // tensor + const_expr -> const_expr
    template <class OpT, class ShapeT1, class DataProviderT1>
    struct overloaded<OpT, tensor_category<ShapeT1, DataProviderT1>, info_const_expr> {
        constexpr overloaded() {}
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            return const_binary_op<OpT, const_coeff<std::decay_t<A>>, B>(OpT(),
                as_const_coeff(forward<A>(a)), forward<B>(b));
        }
    };
    // const_expr + tensor -> const_expr
    template <class OpT, class ShapeT2, class DataProviderT2>
    struct overloaded<OpT, info_const_expr, tensor_category<ShapeT2, DataProviderT2>> {
        constexpr overloaded() {}
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            return const_binary_op<OpT, A, const_coeff<std::decay_t<B>>>(OpT(),
                forward<A>(a), as_const_coeff(forward<B>(b)));
        }
    };
    // tensor + other -> tensor
    template <class OpT, class ShapeT1, class DataProviderT1, class OtherT>
    struct overloaded<OpT, tensor_category<ShapeT1, DataProviderT1>, OtherT> {
        constexpr overloaded() {}
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            return make_tensor(a.shape(), make_tensor_ewise_op_result(
                OpT()(const_symbol<0>(), forward<B>(b)), forward<A>(a)));
        }
    };
    // other + tensor -> tensor
    template <class OpT, class ShapeT2, class DataProviderT2, class OtherT>
    struct overloaded<OpT, OtherT, tensor_category<ShapeT2, DataProviderT2>> {
        constexpr overloaded() {}
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            return make_tensor(b.shape(), make_tensor_ewise_op_result(
                OpT()(forward<A>(a), const_symbol<0>()), forward<B>(b)));
        }
    };


    // ewise_mul
    template <class A, class B, class = std::enable_if_t<
        is_tensor<std::decay_t<A>>::value && is_tensor<std::decay_t<B>>::value>>
    constexpr auto ewise_mul(A && a, B && b) {
        assert(a.shape() == b.shape());
        return make_tensor(a.shape(), make_tensor_ewise_op_result(binary_op_mul(), forward<A>(a), forward<B>(b)));
    }





    // dot product
    template <class C1, class C2, bool RInd1, bool RSub1, bool RInd2, bool RSub2>
    constexpr auto dot(const tensor_readable<C1, RInd1, RSub1, true> & a, const tensor_readable<C2, RInd2, RSub2, true> & b) {
        return ewise_mul(a.category(), b.category()).sum();
    }   




    // reshape
    template <class ShapeT, class DPT, class ToShapeT>
    constexpr tensor_category<ToShapeT, DPT> reshape(const tensor_category<ShapeT, DPT> & t, const ToShapeT & nshape) {
        static_assert(is_tensor_shape<ToShapeT>::value, "invalid type");
        return make_tensor(nshape, t.data_provider());
    }
    template <class ShapeT, class DPT, class ToShapeT>
    constexpr tensor_category<ToShapeT, DPT> reshape(const tensor_category<ShapeT, DPT> && t, const ToShapeT & nshape) {
        static_assert(is_tensor_shape<ToShapeT>::value, "invalid type");
        return make_tensor(nshape, std::move(t.data_provider()));
    }



    // permute
    template <class InputT, class ... IndexTs>
    struct permute_result {
        using value_type = typename std::decay_t<InputT>::value_type;
        constexpr permute_result(InputT && in) 
            : input(forward<InputT>(in)) {}
        InputT input;
        
        template <class Archive>
        void serialize(Archive & ar) {
            ar(input);
        }
    };

    namespace tensor_traits {
        template <class ShapeT, class InputT, class ... IndexTs>
        struct readable_at_subs<tensor_category<ShapeT, permute_result<InputT, IndexTs ...>>> : yes {};

        template <class ShapeT, class InputT, class SubsTupleT, class ... IndexTs, size_t ... Is>
        constexpr decltype(auto) _at_subs_const_impl_seq (
            const tensor_category<ShapeT, permute_result<InputT, IndexTs ...>> & a,
            SubsTupleT && subs, const_ints<size_t, Is...>) {
            return a.data_provider().input.at_subs_const(
                std::get<decltype(find_first_of(cat(IndexTs() ...), const_index<Is>()))::value>(subs) ...);
        }

        template <class ShapeT, class InputT, class ... IndexTs, class ... SubTs>
        constexpr decltype(auto) at_subs_const_impl(
            const tensor_category<ShapeT, permute_result<InputT, IndexTs ...>> & a,
            const SubTs & ... subs) {
            return _at_subs_const_impl_seq(a, std::forward_as_tuple(subs ...), 
                make_const_sequence(const_size<sizeof...(IndexTs)>()));
        }
    }

    template <class InputT, class = std::enable_if_t<is_tensor<std::decay_t<InputT>>::value>, class ... IndexTs>
    constexpr auto permute(InputT && input, const IndexTs & ... indices) {
        static_assert(sizeof...(IndexTs) == input.rank, "invalid index number");
        assert(all_different(indices ...) && "duplicated indices not allowed");
        return make_tensor(permute(input.shape(), indices ...), 
            permute_result<InputT, IndexTs...>(forward<InputT>(input)));
    }




    // cat


    // repeat



    // 


}