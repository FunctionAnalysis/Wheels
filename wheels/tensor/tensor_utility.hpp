#pragma once

#include <cassert>

#include "tensor_shape.hpp"

namespace wheels {

    // default const iterator
    template <class CategoryT>
    struct ts_const_iterator_naive: std::iterator<std::random_access_iterator_tag,
        typename CategoryT::value_type,
        std::ptrdiff_t> {
        const CategoryT & self;
        size_t ind;
        constexpr ts_const_iterator_naive(const CategoryT & s, size_t i = 0) : self(s), ind(i) {}
        constexpr decltype(auto) operator * () const { return self.at_index_const(ind); }
        constexpr decltype(auto) operator -> () const { return self.at_index_const(ind); }
        ts_const_iterator_naive & operator ++() { ++ind; return *this; }
        ts_const_iterator_naive & operator --() { assert(ind != 0);  --ind; return *this; }
        ts_const_iterator_naive & operator +=(size_t s) { ind += s; return *this; }
        ts_const_iterator_naive & operator -=(size_t s) { ind -= s; return *this; }
        constexpr ts_const_iterator_naive operator + (size_t s) const { return ts_const_iterator_naive(self, ind + s); }
        constexpr ts_const_iterator_naive operator - (size_t s) const { return ts_const_iterator_naive(self, ind - s); }
        std::ptrdiff_t operator - (const ts_const_iterator_naive & it) const { return ind - it.ind; }
        constexpr bool operator == (const ts_const_iterator_naive & it) const {
            assert(&self == &(it.self));
            return ind == it.ind;
        }
        constexpr bool operator != (const ts_const_iterator_naive & it) const {
            return ind != it.ind;
        }
        constexpr bool operator < (const ts_const_iterator_naive & it) const {
            return ind < it.ind;
        }
    };


    // default nonconst iterator
    template <class CategoryT>
    struct ts_nonconst_iterator_naive : std::iterator<std::random_access_iterator_tag,
        typename CategoryT::value_type,
        std::ptrdiff_t> {
        CategoryT & self;
        size_t ind;
        constexpr ts_nonconst_iterator_naive(CategoryT & s, size_t i = 0) : self(s), ind(i) {}
        decltype(auto) operator * () const { return self.at_index_nonconst(ind); }
        decltype(auto) operator -> () const { return self.at_index_nonconst(ind); }
        ts_nonconst_iterator_naive & operator ++() { ++ind; return *this; }
        ts_nonconst_iterator_naive & operator --() { assert(ind != 0);  --ind; return *this; }
        ts_nonconst_iterator_naive & operator +=(size_t s) { ind += s; return *this; }
        ts_nonconst_iterator_naive & operator -=(size_t s) { ind -= s; return *this; }
        constexpr ts_nonconst_iterator_naive operator + (size_t s) const { return ts_nonconst_iterator_naive(self, ind + s); }
        constexpr ts_nonconst_iterator_naive operator - (size_t s) const { return ts_nonconst_iterator_naive(self, ind - s); }
        std::ptrdiff_t operator - (const ts_nonconst_iterator_naive & it) const { return ind - it.ind; }
        constexpr bool operator == (const ts_nonconst_iterator_naive & it) const {
            assert(&self == &(it.self));
            return ind == it.ind;
        }
        constexpr bool operator != (const ts_nonconst_iterator_naive & it) const {
            return ind != it.ind;
        }
        constexpr bool operator < (const ts_nonconst_iterator_naive & it) const {
            return ind < it.ind;
        }
    };


    // default nonzero iterator
    template <class CategoryT>
    struct ts_nonzero_iterator_naive : std::iterator<std::forward_iterator_tag,
        typename CategoryT::value_type,
        std::ptrdiff_t> {
        const CategoryT & self;
        size_t ind;
        size_t numel;
        constexpr ts_nonzero_iterator_naive(const CategoryT & s, size_t i = 0) 
            : self(s), numel(s.numel()), ind(i) {}
        
        constexpr decltype(auto) operator * () const { return self.at_index_const(ind); }
        constexpr decltype(auto) operator -> () const { return self.at_index_const(ind); }

        ts_nonzero_iterator_naive & operator ++() { 
            assert(ind != numel);
            ++ind; 
            while (ind != numel && !self.at_index_const(ind)) {
                ++ind;
            }
            return *this; 
        }
        ts_nonzero_iterator_naive & operator --() { 
            assert(ind != 0);  
            --ind; 
            while (ind != 0 && !self.at_index_const(ind)) {
                --ind;
            }
            return *this; 
        }

        constexpr bool operator == (const ts_nonzero_iterator_naive & it) const {
            assert(&self == &(it.self));
            return ind == it.ind;
        }
        constexpr bool operator != (const ts_nonzero_iterator_naive & it) const {
            return ind != it.ind;
        }
        constexpr bool operator < (const ts_nonzero_iterator_naive & it) const {
            return ind < it.ind;
        }
    };




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




}