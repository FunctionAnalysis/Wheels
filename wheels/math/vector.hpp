#pragma once

#include "tensor.hpp"
#include "tensor_op.hpp"

namespace wheels {

    template <class LayoutT>
    class vector_base : public tensor_base<LayoutT> {
    public:
        auto length() const { return norm(); }
        auto normalized() const & { return layout() / length(); }
        auto normalized() && {return std::move(layout()) / length(); }

        constexpr auto n() const { return size(const_index<0>()); }

        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }

        decltype(auto) x() { return at_index_nonconst(0); }
        decltype(auto) y() { return at_index_nonconst(1); }
        decltype(auto) z() { return at_index_nonconst(2); }
        decltype(auto) w() { return at_index_nonconst(3); }

        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }

        decltype(auto) red() { return at_index_nonconst(0); }
        decltype(auto) green() { return at_index_nonconst(1); }
        decltype(auto) blue() { return at_index_nonconst(2); }
        decltype(auto) alpha() { return at_index_nonconst(3); }
    };
    template <class ST, class SizeT, class DPT>
    class tensor_pattern<tensor_layout<tensor_shape<ST, SizeT>, DPT>>
        : public vector_base<tensor_layout<tensor_shape<ST, SizeT>, DPT>> {};

    // is_vector
    template <class T> struct is_vector : no {};
    template <class ST, class SizeT, class DPT>
    struct is_vector<tensor_layout<tensor_shape<ST, SizeT>, DPT>> : yes {};



    // vector functions
    namespace tdp {
        template <class T, size_t Idx>
        struct unit_at {
            using value_type = T;
            constexpr unit_at() {}
            template <class Archiver>
            void serialize(Archiver &) {}
        };
        // accessing elements
        template <class T, size_t Idx> struct is_element_readable_at_index<unit_at<T, Idx>> : yes {};
        template <class T, size_t Idx, class IndexT>
        constexpr T element_at_index(const unit_at<T, Idx> & a, const IndexT & index) {
            return index == Idx ? 1 : 0;
        }
    }

    // unit_x/y/z
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_x(const SizeT & s = const_ints<ST, 3>()) {
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 0>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_y(const SizeT & s = const_ints<ST, 3>()) {
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 1>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_z(const SizeT & s = const_ints<ST, 3>()) {
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 2>());
    }

    // dot
    template <class LT1, class LT2>
    constexpr auto dot(const tensor_base<LT1> & t1, const tensor_base<LT2> & t2) {
        assert(t1.shape() == t2.shape());
        return ewise_mul(t1.layout(), t2.layout()).sum();
    }

    // cross
    template <class LT1, class LT2>
    constexpr vec_<typename LT1::value_type, 3> cross(const vector_base<LT1> & a,
        const vector_base<LT2> & b) {
        assert(a.numel() == 3 && b.numel() == 3);
        return vec_<typename LT1::value_type, 3>(with_elements,
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }

    // angle
    template <class LT1, class LT2>
    constexpr double angle(const vector_base<LT1> & direction1, const vector_base<LT2> & direction2) {
        assert(direction1.numel() == direction.numel());
        return acos(abs(dot(direction1, direction2)));
    }

    

}