#include <gtest/gtest.h>

#include "overloads.hpp"

using namespace wheels;

template <class T> struct A {};

namespace wheels {
    template <class T>
    struct join_overloading<A<T>> : yes {};

    template <class T>
    struct overloaded<unary_op_minus, A<T>> {
        constexpr overloaded() {}
        const char * operator()(A<T> && v) const {
            return "rvalue";
        } 
        const char * operator()(A<T> & v) const {
            return "lvalue";
        }
        const char * operator()(const A<T> & v) const {
            return "const lvalue";
        }
        template <class TT>
        const char * operator()(TT &&) const {
            return "other type";
        }
    };

    template <class T>
    struct overloaded<binary_op_plus, A<T>, int> {
        template <class TT, class II>
        const char * operator()(TT &&, II &&) const {
            return "A<T> + int";
        }
    };
    template <class T>
    struct overloaded<binary_op_plus, int, A<T>> {
        template <class TT, class II>
        const char * operator()(TT &&, II &&) const {
            return "int + A<T>";
        }
    };

}


TEST(core, overloads) {
    A<int> a;
    std::cout << -a << std::endl;
    std::cout << -A<int>() << std::endl;
    int ia = 0;
    auto nia = -ia;

    std::cout << a + 1 << std::endl;
    std::cout << 1 + 1 << std::endl;
    std::cout << 1 + a << std::endl;

}