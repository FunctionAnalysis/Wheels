#pragma once

#include <cassert>

#include "../core/types.hpp"

#include "shape.hpp"

namespace wheels {

    // indexed_iterator_base
    template <class T, class IndexT, class DerivedIterT>
    struct indexed_iterator_base
        : std::iterator<std::random_access_iterator_tag, T> {
        
        constexpr indexed_iterator_base(const IndexT & i, const IndexT & e) 
            : ind(i), end_ind(e) {}

        DerivedIterT & operator ++() { ++ind; return derived(); }
        DerivedIterT & operator --() { assert(ind != 0);  --ind; return derived(); }
        DerivedIterT & operator +=(size_t s) { ind += s; return derived(); }
        DerivedIterT & operator -=(size_t s) { ind -= s; return derived(); }

        DerivedIterT operator + (size_t s) const {
            DerivedIterT iter = derived();
            iter += s;
            return iter;
        }
        DerivedIterT operator - (size_t s) const {
            DerivedIterT iter = derived();
            iter -= s;
            return iter;
        }

        std::ptrdiff_t operator - (const DerivedIterT & it) const { return ind - it.ind; }

        constexpr bool operator == (const DerivedIterT & it) const {
            return ind == it.ind;
        }
        constexpr bool operator != (const DerivedIterT & it) const {
            return ind != it.ind;
        }
        constexpr bool operator < (const DerivedIterT & it) const {
            return ind < it.ind;
        }

        const DerivedIterT & derived() const { return static_cast<const DerivedIterT &>(*this); }
        DerivedIterT & derived() { return static_cast<DerivedIterT &>(*this); }

        IndexT ind;
        IndexT end_ind;
    };



    // get only the nonzero values in iteration
    template <class IterT>
    struct nonzero_iterator_wrapper : std::iterator<std::forward_iterator_tag,
        typename IterT::value_type,
        typename IterT::difference_type,
        typename IterT::pointer,
        typename IterT::reference> {

        using value_type = typename IterT::value_type;

        constexpr nonzero_iterator_wrapper(const IterT & it, const IterT & e)
            : iter(it), end(e) {
            skip_zeros();
        }
        
        constexpr decltype(auto) operator * () const { return * iter; }
        constexpr decltype(auto) operator -> () const { return iter; }

        nonzero_iterator_wrapper & operator ++() {
            assert(iter != end);
            ++iter;
            skip_zeros();
            return *this; 
        }
        constexpr bool operator == (const nonzero_iterator_wrapper & it) const {
            return iter == it.iter;
        }
        constexpr bool operator != (const nonzero_iterator_wrapper & it) const {
            return iter != it.iter;
        }

        void skip_zeros() {
            while (iter != end && *iter == types<value_type>::zero()) {
                ++iter;
            }
        }

        IterT iter;
        IterT end;
    };
    template <class IterT>
    constexpr auto wrap_nonzero_iterator(const IterT & it, const IterT & end) {
        return nonzero_iterator_wrapper<IterT>(it, end);
    }


    // get the value (second) part of a pair in iteration
    template <class IterT>
    struct pair_second_iterator_wrapper : std::iterator<std::forward_iterator_tag,
        typename IterT::value_type::second_type> {
        using value_type = typename IterT::value_type::second_type;

        constexpr pair_second_iterator_wrapper(const IterT & it)
            : iter(it) {}

        constexpr decltype(auto) operator * () const { return iter->second; }
        constexpr decltype(auto) operator -> () const { return &(iter->second); }

        pair_second_iterator_wrapper & operator ++() { ++iter; return *this; }
        constexpr bool operator == (const pair_second_iterator_wrapper & it) const {
            return iter == it.iter;
        }
        constexpr bool operator != (const pair_second_iterator_wrapper & it) const {
            return iter != it.iter;
        }

        IterT iter;
    };
    template <class IterT>
    constexpr auto wrap_pair_second_iterator(const IterT & it) {
        return pair_second_iterator_wrapper<IterT>(it);
    }




    // index_accessible_from_iterator
    // iter2ind
    namespace tensor_traits {

        template <class IterT>
        struct index_accessible_from_iterator : no {};

        template <class IterT>
        struct index_accessible_from_iterator<nonzero_iterator_wrapper<IterT>>
            : index_accessible_from_iterator<IterT> {};
        template <class IterT>
        constexpr size_t iter2ind(const nonzero_iterator_wrapper<IterT> & iter) {
            return iter2ind(iter.iter);
        }

        template <class IterT>
        struct index_accessible_from_iterator<pair_second_iterator_wrapper<IterT>> : yes {};
        template <class IterT>
        constexpr size_t iter2ind(const pair_second_iterator_wrapper<IterT> & iter) {
            return iter.iter->first;
        }

    }


}