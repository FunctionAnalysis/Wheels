#pragma once

#include <iostream>

#include "tensor.hpp"
#include "tensor_functions.hpp"

namespace wheels {

    // special shapes
    // vector shape
    template <class CategoryT, bool Writable>
    class vector : public ts_specific_shape_base<CategoryT> {
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
    template <class CategoryT, class ST, class SizeT>
    class ts_specific_shape<CategoryT, tensor_shape<ST, SizeT>>
        : public vector<CategoryT, ts_traits::writable<CategoryT>::value> {};

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
    class matrix : public ts_specific_shape_base<CategoryT> {
    public:
        constexpr auto rows() const { return size(const_index<0>()); }
        constexpr auto cols() const { return size(const_index<1>()); }
        constexpr auto area() const { return rows() * cols(); }
    };
    template <class CategoryT, class ST, class MT, class NT>
    class ts_specific_shape<CategoryT, tensor_shape<ST, MT, NT>>
        : public matrix<CategoryT, ts_traits::writable<CategoryT>::value> {};

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
    class cube : public ts_specific_shape_base<CategoryT> {
    public:
        constexpr auto volume() const {
            return size(const_index<0>()) * size(const_index<1>()) * size(const_index<2>()); 
        }
    };
    template <class CategoryT, class ST, class N1T, class N2T, class N3T>
    class ts_specific_shape<CategoryT, tensor_shape<ST, N1T, N2T, N3T>>
        : public cube<CategoryT, ts_traits::writable<CategoryT>::value> {};







    // special value types
    template <class CategoryT>
    class boolean_tensor : public ts_specific_value_type_base<CategoryT> {
    public:
        // conversion to bool value
        constexpr operator bool() const { return all(); }
    };

    template <class CategoryT>
    class ts_specific_value_type<CategoryT, bool> : public boolean_tensor<CategoryT> {};



}
