#pragma once

#include <iterator>

namespace wheels {

    template <class T, class IterT, class FunT>
    class transform_iterator : public std::iterator<
        typename std::iterator_traits<IterT>::iterator_category, T,
        typename std::iterator_traits<IterT>::difference_type> {
    public:
        using difference_type = typename std::iterator_traits<IterT>::difference_type;

        constexpr transform_iterator() : current(), fun() {}
        template <class F>
        constexpr transform_iterator(IterT it, F && f) : current(it), fun(forward<F>(f)) {}
        constexpr IterT base() const { return current; }

        decltype(auto) operator*() const { return fun(*current); }
        decltype(auto) operator->() const { return &(**this); }

        transform_iterator & operator ++() { ++current; return *this; }
        transform_iterator & operator ++(int) { auto tmp = *this; ++current; return tmp; }
        transform_iterator & operator --() { --current; return *this; }
        transform_iterator & operator --(int) { auto tmp = *this; --current; return tmp; }

        transform_iterator& operator+=(difference_type off) {	// increment by integer
            current += off;
            return (*this);
        }
        transform_iterator operator+(difference_type off) const {	// return this + integer
            return (transform_iterator(current + off, fun));
        }

        transform_iterator& operator-=(difference_type off) {	// decrement by integer
            current -= off;
            return (*this);
        }
        transform_iterator operator-(difference_type off) const {	// return this - integer
            return (transform_iterator(current - off, fun));
        }
    protected:
        IterT current;
        FunT fun;
    };

    template <class T, class IterT, class FunT>
    constexpr bool operator == (const transform_iterator<T, IterT, FunT> & i1, const transform_iterator<T, IterT, FunT> & i2) {
        return i1.base() == i2.base();
    }
    template <class T, class IterT, class FunT>
    constexpr bool operator != (const transform_iterator<T, IterT, FunT> & i1, const transform_iterator<T, IterT, FunT> & i2) {
        return !(i1 == i2);
    }
    template <class T, class IterT, class FunT>
    constexpr bool operator < (const transform_iterator<T, IterT, FunT> & i1, const transform_iterator<T, IterT, FunT> & i2) {
        return i1.base() < i2.base();
    }

    template <class T, class IterT, class FunT>
    constexpr auto operator - (const transform_iterator<T, IterT, FunT> & i1, const transform_iterator<T, IterT, FunT> & i2) {
        return i1.base() - i2.base();
    }


    // make_transform_iterator
    template <class IterT, class FunT>
    constexpr auto make_transform_iterator(IterT it, FunT f) {
        using value_t = std::decay_t<decltype(f(*it))>;
        return transform_iterator<value_t, IterT, FunT>(it, f);
    }

}
