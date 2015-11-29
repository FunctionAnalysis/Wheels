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
        constexpr decltype(auto) operator -> () const { return &(self.at_index_const(ind)); }
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
        decltype(auto) operator -> () const { return &(self.at_index_nonconst(ind)); }
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


    // nonzero iterator wrapper
    template <class IterT>
    struct nonzero_iterator_of : std::iterator<std::forward_iterator_tag,
        typename IterT::value_type,
        typename IterT::difference_type,
        typename IterT::pointer,
        typename IterT::reference> {
        IterT iter;
        IterT end;
        constexpr nonzero_iterator_of(const IterT & it, const IterT & e)
            : iter(it), end(e) {}
        
        constexpr decltype(auto) operator * () const { return * iter; }
        constexpr decltype(auto) operator -> () const { return iter; }

        nonzero_iterator_of & operator ++() {
            assert(iter != end);
            ++iter;
            while (iter != end && !*iter) {
                ++iter;
            }
            return *this; 
        }
        constexpr bool operator == (const nonzero_iterator_of & it) const {
            return iter == it.iter;
        }
        constexpr bool operator != (const nonzero_iterator_of & it) const {
            return iter != it.iter;
        }
    };


    // second in pair iterator wrapper
    template <class IterT>
    struct second_in_pair_iterator_of : std::iterator<std::forward_iterator_tag,
        typename IterT::value_type::second_type> {
        IterT iter;
        constexpr second_in_pair_iterator_of(const IterT & it)
            : iter(it) {}

        constexpr decltype(auto) operator * () const { return iter->second; }
        constexpr decltype(auto) operator -> () const { return &(iter->second); }

        second_in_pair_iterator_of & operator ++() { ++iter; return *this; }
        constexpr bool operator == (const second_in_pair_iterator_of & it) const {
            return iter == it.iter;
        }
        constexpr bool operator != (const second_in_pair_iterator_of & it) const {
            return iter != it.iter;
        }
    };


    // index_accessible_from_iterator
    // iter2ind
    namespace ts_traits {

        template <class IterT>
        struct index_accessible_from_iterator : no {};

        template <class CategoryT>
        struct index_accessible_from_iterator<ts_const_iterator_naive<CategoryT>> : yes {};
        template <class CategoryT>
        constexpr size_t iter2ind(const ts_const_iterator_naive<CategoryT> & iter) {
            return iter.ind;
        }

        template <class CategoryT>
        struct index_accessible_from_iterator<ts_nonconst_iterator_naive<CategoryT>> : yes {};
        template <class CategoryT>
        constexpr size_t iter2ind(const ts_nonconst_iterator_naive<CategoryT> & iter) {
            return iter.ind;
        }

        template <class IterT>
        struct index_accessible_from_iterator<nonzero_iterator_of<IterT>>
            : index_accessible_from_iterator<IterT> {};
        template <class IterT>
        constexpr size_t iter2ind(const nonzero_iterator_of<IterT> & iter) {
            return iter2ind(iter.iter);
        }

        template <class IterT>
        struct index_accessible_from_iterator<second_in_pair_iterator_of<IterT>> : yes {};
        template <class IterT>
        constexpr size_t iter2ind(const second_in_pair_iterator_of<IterT> & iter) {
            return iter.iter->first;
        }

    }





    namespace index_tags {
        constexpr auto first = const_index<0>();
        constexpr auto length = const_symbol<0>();
        const auto last = length - const_index<1>();
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