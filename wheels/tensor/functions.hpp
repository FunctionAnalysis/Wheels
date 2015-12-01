#pragma once

#include <iostream>

#include "../core/overloads.hpp"

#include "categories.hpp"

namespace wheels {


    // constants
    template <class T>
    class constant {
    public:
        using value_type = T;
        T val;
        template <class TT>
        constexpr constant(TT && v) : val(forward<TT>(v)) {}
        template <class ... ArgTs>
        constexpr const T & operator()(ArgTs && ...) const { return val; }
        template <class Archive> void serialize(Archive & ar) { ar(val); }
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
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return make_tensor(make_shape<ST>(sizes...), constant<T>(types<T>::zero()));
    }

    // ones
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs ...> & shape) {
        return make_tensor(shape, constant<T>(1));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return make_tensor(make_shape<ST>(sizes...), constant<T>(1));
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
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto meshgrid(const SizeTs & ... sizes) {
        return details::_meshgrid_seq<T>(make_shape(sizes ...), make_const_sequence(const_size<sizeof...(SizeTs)>()));
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
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_x(const SizeT & s = SizeT()) {
        return make_tensor(make_shape<ST>(s), unit_axis_result<T, 0>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_y(const SizeT & s = SizeT()) {
        return make_tensor(make_shape<ST>(s), unit_axis_result<T, 1>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_z(const SizeT & s = SizeT()) {
        return make_tensor(make_shape<ST>(s), unit_axis_result<T, 2>());
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
        : constant_value_iterator_base<T, eye_result_nonzero_iterator<ShapeT, T>> {
        const ShapeT & shape;
        constexpr eye_result_nonzero_iterator(size_t ind, const T & val, const ShapeT & s)
            : constant_value_iterator_base<T, eye_result_nonzero_iterator>(ind, min_shape_size(s), val),
            shape(s) {}
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
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto eye(const SizeTs & ... sizes) {
        return make_tensor(make_shape<ST>(sizes ...), eye_result<T>());
    }




    
    // ewise op
    template <class T, class OpT, class ... InputTs>
    struct ewise_op_result {
        using value_type = T;
        using op_type = OpT;
        OpT op;
        std::tuple<InputTs ...> inputs;
        
        template <class OpTT, class ... InputTTs>
        constexpr explicit ewise_op_result(OpTT && o, InputTTs && ... ins)
            : op(forward<OpTT>(o)), inputs(std::forward<InputTTs>(ins) ...) {}
        
        template <class Archive> 
        void serialize(Archive & ar) { 
            ar(op, inputs); 
        }
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
                OpT()(const_symbol<0>(), forward<A>(a)), forward<B>(b)));
        }
    };







    // special shapes
    // vector shape
    template <class CategoryT, bool Writable>
    class vector : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }
        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }
        constexpr auto normalized() const { return category() / norm(); }
    };
    template <class CategoryT>
    class vector<CategoryT, true> : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }
        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }
        decltype(auto) x() { return at_index_nonconst(0); }
        decltype(auto) y() { return at_index_nonconst(1); }
        decltype(auto) z() { return at_index_nonconst(2); }
        decltype(auto) w() { return at_index_nonconst(3); }
        decltype(auto) red() { return at_index_nonconst(0); }
        decltype(auto) green() { return at_index_nonconst(1); }
        decltype(auto) blue() { return at_index_nonconst(2); }
        decltype(auto) alpha() { return at_index_nonconst(3); }
        constexpr auto normalized() const { return category() / norm(); }
    };
    template <class CategoryT, class ST, class SizeT>
    class tensor_specific_shape<CategoryT, tensor_shape<ST, SizeT>>
        : public vector<CategoryT, tensor_traits::writable<CategoryT>::value> {};

    // cross
    template <class CategoryT1, class CategoryT2, bool W1, bool W2>
    constexpr auto cross(const vector<CategoryT1, W1> & a, const vector<CategoryT2, W2> & b) {
        using value_t = std::common_type_t<typename CategoryT1::value_type, typename CategoryT2::value_type>
            assert(a.numel() == 3 && b.numel() == 3);
        return vec_<value_t, 3>(with_elements,
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }

    // print
    template <class CategoryT, bool Writable>
    inline std::ostream & operator << (std::ostream & os, const vector<CategoryT, Writable> & v) {
        os << "[";
        for (size_t ind = 0; ind < v.numel() - 1; ind++) {
            os << v[ind] << ", ";
        }
        os << v[index_tags::last] << "]";
        return os;
    }






    // matrix shape
    template <class CategoryT, bool Writable>
    class matrix : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr auto rows() const { return size(const_index<0>()); }
        constexpr auto cols() const { return size(const_index<1>()); }
    };
    template <class CategoryT, class ST, class MT, class NT>
    class tensor_specific_shape<CategoryT, tensor_shape<ST, MT, NT>>
        : public matrix<CategoryT, tensor_traits::writable<CategoryT>::value> {};

    // matrix mul
    

    // print
    template <class CategoryT, bool Writable>
    inline std::ostream & operator << (std::ostream & os, const matrix<CategoryT, Writable> & m) {
        for (size_t r = 0; r < m.rows(); r++) {
            os << "[";
            for (size_t c = 0; c < m.cols() - 1; c++) {
                os << m(r, c) << ", ";
            }
            os << m(r, index_tags::last) << "]\n";
        }
        return os;
    }



    // cube shape
    template <class CategoryT, bool Writable>
    class cube : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr auto volume() const {
            return size(const_index<0>()) * size(const_index<1>()) * size(const_index<2>());
        }
    };
    template <class CategoryT, class ST, class N1T, class N2T, class N3T>
    class tensor_specific_shape<CategoryT, tensor_shape<ST, N1T, N2T, N3T>>
        : public cube<CategoryT, tensor_traits::writable<CategoryT>::value> {};







    // special value types
    template <class CategoryT>
    class boolean_tensor : public tensor_specific_value_type_base<CategoryT> {
    public:
        // conversion to bool value
        constexpr operator bool() const { return all(); }
    };
    template <class CategoryT>
    class tensor_specific_value_type<CategoryT, bool>
        : public boolean_tensor<CategoryT> {};







    // dot product
    template <class C1, class C2, bool RInd1, bool RSub1, bool RInd2, bool RSub2>
    constexpr auto dot(const tensor_readable<C1, RInd1, RSub1, true> & a,
        const tensor_readable<C2, RInd2, RSub2, true> & b) {
        return (a.category() * b.category()).sum();
    }   




    // reshape





    // cat


    // repeat


    // permute


    // 


}