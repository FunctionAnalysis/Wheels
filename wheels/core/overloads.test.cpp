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


template <class T> 
struct B : wheels::object_overloadings<B<T>, 
    wheels::member_op_bracket,
    wheels::member_op_paren> {};

namespace wheels {
    template <class T>
    struct overloaded<member_op_bracket, B<T>, int> {
        template <class TT, class II>
        constexpr const char * operator()(TT &&, II &&) const {
            return "B<T>[int]";
        }
    };
    template <class T>
    struct overloaded<member_op_paren, B<T>, int> {
        template <class TT, class II>
        constexpr const char * operator()(TT &&, II &&) const {
            return "B<T>(int)";
        }
    };
    //template <class T>
    //struct overloaded<member_op_increment, B<T>> {
    //    template <class TT>
    //    const char * operator()(TT &&) const {
    //        return "++ B<T>";
    //    }
    //};
    //template <class T>
    //struct overloaded<member_op_increment, B<T>, int> {
    //    template <class TT>
    //    const char * operator()(TT &&, int) const {
    //        return "B<T> ++";
    //    }
    //};
}

TEST(core, member_overloads) {
    B<int> bb;
    std::cout << bb[1] << std::endl;
    std::cout << bb(1) << std::endl;
    std::cout << B<int>()(1) << std::endl;
    
    std::is_standard_layout<B<int>>::value;

    auto c = bb[1];

    int i = 1;
    
}
