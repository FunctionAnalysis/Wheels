#pragma once

#include <cassert>

#include "../core/types.hpp"
#include "../core/const_expr.hpp"
#include "../core/overloads.hpp"
#include "../core/serialize.hpp"

#include "shape.hpp"

namespace wheels {

    // instantiate category_for_overloading<T> as category_tensor<...> to join tensor ops
    template <class ShapeT, class EleT, class T> struct category_tensor {};

    // base of all tensor types
    template <class T> 
    struct tensor_base {
        constexpr const T & derived() const { return static_cast<const T &>(*this); }
        T & derived() { return static_cast<T &>(*this); }
    };

    namespace details {
        template <class T>
        using _shape_t = typename std::decay_t<T>::shape_type;
        template <class T>
        using _element_t = typename std::decay_t<T>::value_type;
    }

    // category_for_overloading
    template <class T, class OpT, class = std::enable_if_t<!std::is_same<OpT, func_fields>::value>>
    constexpr auto category_for_overloading(const tensor_base<T> &, const OpT &) {
        return types<category_tensor<details::_shape_t<T>, details::_element_t<T>, T>>();
    }



    // -- necessary tensor functions
    // Shape shape_of(ts);
    template <class T>
    constexpr tensor_shape<size_t> shape_of(const tensor_base<T> &) {
        static_assert(always<bool, false, T>::value, "shape_of(const T &) is not supported here");
    }

    // Scalar element_at(ts, subs ...);
    template <class T, class ... SubTs>
    constexpr double element_at(const tensor_base<T> & t, const SubTs & ...) {
        static_assert(always<bool, false, T>::value, "element_at(const T &) is not supported here");
    }
    template <class T, class ... SubTs>
    constexpr double & element_at(tensor_base<T> & t, const SubTs & ...) {
        static_assert(always<bool, false, T>::value, "element_at(T &) is not supported here");
    }
    // member paren op is for element_at
    template <class ShapeT, class EleT, class T, class ... SubTs>
    struct overloaded<member_op_paren, category_tensor<ShapeT, EleT, T>, SubTs ...> {
        template <class CallerT, class ... TTs>
        decltype(auto) operator()(CallerT && caller, TTs && ... subs) const {
            return element_at(forward<CallerT>(caller), subs ...);
        }
    };


    // -- auxiliary tensor functions
    // auto rank_of(ts)
    template <class T>
    constexpr auto rank_of(const tensor_base<T> & t) {
        return const_size<decltype(shape_of(t.derived()))::rank>();
    }

    // auto size_at(ts, const_int);
    template <class T, class ST, ST Idx>
    constexpr auto size_at(const tensor_base<T> & t, const const_ints<ST, Idx> & idx) {
        return shape_of(t.derived()).at(idx);
    }

    // auto numel(ts)
    template <class T>
    constexpr auto numel(const tensor_base<T> & t) {
        return shape_of(t.derived()).magnitude();
    }

    // Scalar element_at_index(ts, index);
    template <class T, class IndexT>
    constexpr decltype(auto) element_at_index(const tensor_base<T> & t, const IndexT & ind) {
        return invoke_with_subs(shape_of(t),
            [&t](auto && ... subs) {return element_at(t.derived(), subs ...); });
    }

    // member bracket op is for element at index
    template <class ShapeT, class EleT, class T, class IndexT>
    struct overloaded<member_op_bracket, category_tensor<ShapeT, EleT, T>, IndexT> {
        template <class CallerT, class TT>
        constexpr decltype(auto) operator()(CallerT && caller, TT && index) const {
            return element_at_index(forward<CallerT>(caller), forward<TT>(index));
        }
    };

    // void reserve_shape(ts, shape);
    template <class T, class ST, class ... SizeTs>
    void reserve_shape(tensor_base<T> &, const tensor_shape<ST, SizeTs...> & shape) {}

    // void for_each_element(ts, functor);
    template <class FunT, class T, class ... Ts>
    void for_each_element(FunT && fun, T && t, Ts && ... ts) {
        assert(all_same(shape_of(t), shape_of(ts) ...));
        for_each_subscript(shape_of(t), [&](auto && ... subs) {
            fun(element_at(t, subs ...), element_at(ts, subs ...) ...);
        });
    }

    // void assign_elements(to, from);
    template <class To, class From>
    void assign_elements(To & to, const From & from) {
        decltype(auto) s = shape_of(from);
        if (shape_of(to) != s) {
            reserve_shape(to, s);
        }
        for_each_element([](auto & to_e, const auto & from_e) {
            to_e = from_e;
        }, to, from);
    }

    // Scalar reduce_elements(ts, initial, functor);
    template <class T, class E, class ReduceT>
    E reduce_elements(const T & t, E initial, ReduceT && red) {
        for_each_element([&initial, &red](auto && e) {initial = red(initial, e); }, 
            t);
        return initial;
    }

    // Scalar dot_product(ts1, ts2);
    template <class T1, class T2>
    decltype(auto) dot_product(const tensor_base<T1> & t1, const tensor_base<T2> & t2) {
        using result_t = std::common_type_t<details::_element_t<T1>, details::_element_t<T2>>;
        assert(shape_of(t1) == shape_of(t2));
        result_t result = 0.0;
        for_each_element([&result](auto && e1, auto && e2) {result += e1 * e2; }, 
            t1.derived(), t2.derived());
        return result;
    }

    // Scalar norm_squared(ts)
    template <class T>
    details::_element_t<T> norm_squared(const tensor_base<T> & t) {
        using result_t = details::_element_t<T>;
        result_t result = 0.0;
        for_each_element([&result](auto && e) {result += e * e; },
            t.derived());
        return result;
    }

    // Scalar norm(ts)
    template <class T>
    constexpr details::_element_t<T> norm(const tensor_base<T> & t) {
        return sqrt(norm_squared(t));
    }



    // -- special tensor functions
    //// ewise_ops
    template <class ShapeT, class EleT, class OpT, class InputT, class ... InputTs>
    class ewise_op_result : public tensor_base<ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs ...>> {
    public:
        using shape_type = ShapeT;
        using value_type = EleT;
        template <class OpTT>
        constexpr explicit ewise_op_result(OpTT && o, InputT && in, InputTs && ... ins)
            : op(forward<OpTT>(o)), inputs(forward<InputT>(in), forward<InputTs>(ins) ...) {}
        template <wheels_enable_if((std::is_same<EleT, bool>::value))>
        constexpr operator bool() const {
            return reduce_elements(*this, true, binary_op_and());
        }
    public:
        template <class V>
        decltype(auto) fields(V && visitor) {
            return visitor(op, inputs);
        }
        template <class V>
        decltype(auto) fields(V && visitor) const {
            return visitor(op, inputs);
        }
    public:
        OpT op;
        std::tuple<InputT, InputTs ...> inputs;
    };
    template <class OpT, class InputT, class ... InputTs>
    constexpr auto make_ewise_op_result(OpT && op, InputT && input, InputTs && ... inputs) {
        using shape_t = details::_shape_t<InputT>;
        using ele_t = std::decay_t<decltype(op(element_at_index(input, 0), element_at_index(inputs, 0) ...))>;
        return ewise_op_result<shape_t, ele_t, std::decay_t<OpT>, InputT, InputTs...>(forward<OpT>(op),
            forward<InputT>(input),
            forward<InputTs>(inputs) ...);
    }
    // shape_of
    template <class ShapeT, class EleT, class OpT, class InputT, class ... InputTs>
    constexpr decltype(auto) shape_of(const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> & ts) {
        return shape_of(std::get<0>(ts.inputs));
    }
    // element_at
    namespace details {
        template <class EwiseOpResultT, size_t ... Is, class ... SubTs>
        constexpr decltype(auto) _element_at_ewise_op_result_seq(
            EwiseOpResultT & ts,
            const const_ints<size_t, Is...> &, const SubTs & ... subs) {
            return ts.op(element_at(std::get<Is>(ts.inputs), subs ...) ...);
        }
    }
    template <class ShapeT, class EleT, class OpT, class InputT, class ... InputTs, class ... SubTs>
    constexpr decltype(auto) element_at(const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> & ts, const SubTs & ... subs) {
        return details::_element_at_ewise_op_result_seq(ts, make_const_sequence_for<InputT, InputTs ...>(), subs ...);
    } 
    // shortcuts
    namespace details {
        template <class EwiseOpResultT, size_t ... Is, class IndexT>
        constexpr decltype(auto) _element_at_index_ewise_op_result_seq(
            EwiseOpResultT & ts,
            const const_ints<size_t, Is...> &, const IndexT & index) {
            return ts.op(element_at_index(std::get<Is>(ts.inputs), index) ...);
        }
    }
    template <class ShapeT, class EleT, class OpT, class InputT, class ... InputTs, class IndexT>
    constexpr decltype(auto) element_at_index(const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> & ts, const IndexT & index) {
        return details::_element_at_index_ewise_op_result_seq(ts, make_const_sequence_for<InputT, InputTs ...>(), index);
    }


    // other ops are overloaded as ewise operation results defaultly
    // all tensors
    template <class OpT, class ShapeT, class EleT, class T, class ... ShapeTs, class ... EleTs, class ... Ts>
    struct overloaded<OpT, category_tensor<ShapeT, EleT, T>, category_tensor<ShapeTs, EleTs, Ts> ...> {
        template <class TT, class ... TTs>
        constexpr decltype(auto) operator()(TT && t, TTs && ... ts) const {
            assert(all_same(shape_of(t), shape_of(ts) ...));         
            return make_ewise_op_result(OpT(), forward<TT>(t), forward<TTs>(ts) ...);
        }
    };
    // tensor vs scalar
    template <class OpT, class ShapeT, class EleT, class T, class ScalarT>
    struct overloaded<OpT, category_tensor<ShapeT, EleT, T>, ScalarT> {
        template <class T1, class T2>
        constexpr decltype(auto) operator()(T1 && t1, T2 && t2) const {
            return make_ewise_op_result(OpT()(const_symbol<0>(), forward<T2>(t2)), forward<T1>(t1));
        }
    };
    // scalar vs tensor
    template <class OpT, class ScalarT, class ShapeT, class EleT, class T>
    struct overloaded<OpT, ScalarT, category_tensor<ShapeT, EleT, T>> {
        template <class T1, class T2>
        constexpr decltype(auto) operator()(T1 && t1, T2 && t2) const {
            return make_ewise_op_result(OpT()(forward<T1>(t1), const_symbol<0>()), forward<T2>(t2));
        }
    };
    // tensor vs const_expr
    template <class OpT, class ShapeT, class EleT, class T>
    struct overloaded<OpT, category_tensor<ShapeT, EleT, T>, category_const_expr> {
        template <class T1, class T2>
        constexpr decltype(auto) operator()(T1 && t1, T2 && t2) const {
            return const_binary_op<OpT, const_coeff<std::decay_t<T1>>, T2>(OpT(),
                as_const_coeff(forward<T1>(t1)), forward<T2>(t2));
        }
    };
    // const_expr vs tensor
    template <class OpT, class ShapeT, class EleT, class T>
    struct overloaded<OpT, category_const_expr, category_tensor<ShapeT, EleT, T>> {
        template <class T1, class T2>
        constexpr decltype(auto) operator()(T1 && t1, T2 && t2) const {
            return const_binary_op<OpT, T1, const_coeff<std::decay_t<T2>>>(OpT(),
                forward<T1>(t1), as_const_coeff(forward<T2>(t2)));
        }
    };


    // auto normalize(ts)
    template <class T>
    constexpr auto normalize(T && t) {
        return forward<T>(t) / norm(t);
    }


    // auto matrix_mul(ts1, ts2);
    template <class EleT, class A, class B, bool AIsMat, bool BIsMat> 
    class matrix_mul_result;
    // matrix + matrix -> matrix
    template <class EleT, class A, class B>
    class matrix_mul_result<EleT, A, B, true, true> : public tensor_base<matrix_mul_result<EleT, A, B, true, true>> {
    public:
        using value_type = EleT;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
        constexpr auto shape() const {
            return make_shape(size_at(_a, const_index<0>()), size_at(b, const_index<1>()));
        }
        using shape_type = decltype(std::declval<matrix_mul_result>().shape());
        template <class SubT1, class SubT2>
        decltype(auto) at_subs(const SubT1 & s1, const SubT2 & s2) const {
            using result_t = std::common_type_t<details::_element_t<A>, details::_element_t<B>>;
            result_t result = types<result_t>::zero();
            for (size_t i = 0; i < size_at(_a, const_index<1>()); i++) {
                result += element_at(_a, s1, i) * element_at(_b, i, s2);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };
    // matrix + vector -> vector
    template <class EleT, class A, class B>
    class matrix_mul_result<EleT, A, B, true, false> : public tensor_base<matrix_mul_result<EleT, A, B, true, false>> {
    public:
        using value_type = EleT;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
        constexpr auto shape() const {
            return make_shape(size_at(_a, const_index<0>()));
        }
        using shape_type = decltype(std::declval<matrix_mul_result>().shape());
        template <class SubT>
        decltype(auto) at_subs(const SubT & s) const {
            using result_t = std::common_type_t<details::_element_t<A>, details::_element_t<B>>;
            result_t result = types<result_t>::zero();
            for (size_t i = 0; i < size_at(_a, const_index<1>()); i++) {
                result += element_at(_a, s, i) * element_at(_b, i);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };
    // vector + matrix -> vector
    template <class EleT, class A, class B>
    class matrix_mul_result<EleT, A, B, false, true> : public tensor_base<matrix_mul_result<EleT, A, B, false, true>> {
    public:
        using value_type = EleT;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
        constexpr auto shape() const {
            return make_shape(size_at(_b, const_index<1>()));
        }
        using shape_type = decltype(std::declval<matrix_mul_result>().shape());
        template <class SubT>
        decltype(auto) at_subs(const SubT & s) const {
            using result_t = std::common_type_t<details::_element_t<A>, details::_element_t<B>>;
            result_t result = types<result_t>::zero();
            for (size_t i = 0; i < size_at(_a, const_index<0>()); i++) {
                result += element_at(_a, i) * element_at(_b, i, s);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };
    
    // shape_of
    template <class EleT, class A, class B, bool AIsMat, bool BIsMat>
    constexpr auto shape_of(const matrix_mul_result<EleT, A, B, AIsMat, BIsMat> & m) {
        return m.shape();
    }
    // element_at
    template <class EleT, class A, class B, bool AIsMat, bool BIsMat, class ... SubTs>
    constexpr decltype(auto) element_at(const matrix_mul_result<EleT, A, B, AIsMat, BIsMat> & m, const SubTs & ... subs) {
        return m.at_subs(subs ...);
    }



    template <class ST1, class MT1, class NT1, class E1, class T1,
        class ST2, class MT2, class NT2, class E2, class T2>
    struct overloaded<binary_op_mul,
        category_tensor<tensor_shape<ST1, MT1, NT1>, E1, T1>,
        category_tensor<tensor_shape<ST2, MT2, NT2>, E2, T2>> {
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
            return matrix_mul_result<std::common_type_t<E1, E2>, A, B, true, true>(forward<A>(a), forward<B>(b));
        }
    };

    template <class ST1, class MT1, class NT1, class E1, class T1,
        class ST2, class MT2, class E2, class T2>
    struct overloaded<binary_op_mul,
        category_tensor<tensor_shape<ST1, MT1, NT1>, E1, T1>,
        category_tensor<tensor_shape<ST2, MT2>, E2, T2>> {
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
            return matrix_mul_result<std::common_type_t<E1, E2>, A, B, true, false>(forward<A>(a), forward<B>(b));
        }
    };

    template <class ST1, class MT1, class E1, class T1,
        class ST2, class MT2, class NT2, class E2, class T2>
    struct overloaded<binary_op_mul,
        category_tensor<tensor_shape<ST1, MT1>, E1, T1>,
        category_tensor<tensor_shape<ST2, MT2, NT2>, E2, T2>> {
        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(size_at(a, const_index<0>()) == size_at(b, const_index<0>()));
            return matrix_mul_result<std::common_type_t<E1, E2>, A, B, false, true>(forward<A>(a), forward<B>(b));
        }
    };




    // transpose




    // auto cross(ts1, ts2);
    /*template <class A, class B>
    constexpr auto cross(const A & a, const B & b) {
        using result_t = std::common_type_t<details::_element_t<A>, details::_element_t<B>>;
        return vec_<value_t, 3>(with_elements,
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }*/





    struct _with_elements {} with_elements;
    template <class T, size_t ... Ss>
    class tensor : public tensor_base<tensor<T, Ss...>>,
        public serializable<tensor<T, Ss ...>> {

    public:
        using value_type = T;
        using shape_type = tensor_shape<size_t, const_size<Ss>...>;

        constexpr tensor() {}

        template <class ... EleTs>
        constexpr tensor(const _with_elements &, EleTs && ... eles)
            : _data{ { (T)forward<EleTs>(eles) ... } } {}

        template <class AnotherT>
        constexpr tensor(const tensor_base<AnotherT> & another) {
            assign_elements(*this, another.derived());
        }
        template <class AnotherT>
        tensor & operator = (const tensor_base<AnotherT> & another) {
            assign_elements(*this, another.derived());
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
            return visitor(_data);
        }
        template <class V>
        constexpr decltype(auto) fields(V && visitor) const {
            return visitor(_data);
        }

    private:
        std::array<T, shape_type::static_magnitude> _data;
    };


    // necessary
    template <class T, size_t ... Ss>
    constexpr auto shape_of(const tensor<T, Ss ...> & t) {
        return t.shape();
    }
    template <class T, size_t ... Ss, class ... SubTs>
    constexpr const T & element_at(const tensor<T, Ss...> & t, const SubTs & ... subs) {
        return t(subs ...);
    }
    template <class T, size_t ... Ss, class ... SubTs>
    T & element_at(tensor<T, Ss...> & t, const SubTs & ... subs) {
        return t(subs ...);
    }

    // auxiliary
    template <class T, size_t ... Ss, class IndexT>
    constexpr const T & element_at_index(const tensor<T, Ss...> & t, const IndexT & ind) {
        return t[ind];
    }
    template <class T, size_t ... Ss, class IndexT>
    T & element_at_index(tensor<T, Ss...> & t, const IndexT & ind) {
        return t[ind];
    }

    template <class T, size_t ... Ss, class ST, class ... SizeTs>
    void reserve_shape(tensor<T, Ss...> & t, const tensor_shape<ST, SizeTs...> & shape) {
        assert(t.shape() == shape);
    }

    template <class FunT, class T, size_t ... Ss, class ... Ts>
    void for_each_element(FunT && fun, const tensor<T, Ss...> & t, Ts && ... ts) {
        assert(all_same(shape_of(t), shape_of(ts) ...));
        for (size_t i = 0; i < numel(t); i++) {
            fun(element_at_index(t, i), element_at_index(ts, i), ...);
        }
    }
    template <class FunT, class T, size_t ... Ss, class ... Ts>
    void for_each_element(FunT && fun, tensor<T, Ss...> & t, Ts && ... ts) {
        assert(all_same(shape_of(t), shape_of(ts) ...));
        for (size_t i = 0; i < numel(t); i++) {
            fun(element_at_index(t, i), element_at_index(ts, i) ...);
        }
    }


    


}