#include <gtest/gtest.h>

#include "fields.hpp"

struct A {
    int a, b;
    auto tuple() {
        return std::forward_as_tuple(a, b);
    }
};

bool operator == (const A & a1, const A & a2) {
    return a1.a == a2.a && a1.b == a2.b;
}

TEST(core, ref_behavior) {

    std::vector<A> as(100), bs(100);
    for (int i = 0; i < as.size(); i++) {
        as[i].a = i;
        as[i].b = -i;
        bs[i].a = -i;
        bs[i].b = i;
    }

    std::vector<std::tuple<int &, int &>> ats;
    for (int i = 0; i < as.size(); i++) {
        ats.push_back(as[i].tuple());
    }

    std::vector<std::tuple<int &, int &>> bts;
    for (int i = 0; i < bs.size(); i++) {
        bts.push_back(bs[i].tuple());
    }   

    ASSERT_FALSE(ats == bts);
    ASSERT_FALSE(as == bs);

    bts = ats;
    
    ASSERT_TRUE(ats == bts);
    ASSERT_TRUE(as == bs);

}



using namespace wheels;

//template <class T1, class T2>
//struct B {
//    template <class U, class V>
//    auto fields(U &&, V && v) {
//        return v(v1, v2);
//    }
//    T1 v1;
//    T2 v2;
//};

//namespace wheels {
//    template <class T1, class T2, class U, class V>
//    auto fields(B<T1, T2> & b, U &&, V && v) {
//        return v(b.v1, b.v2);
//    }
//}

TEST(core, simple_visit) {

    has_global_func_fields<char, int, field_visitor<pack_as_tuple, visit_to_tuplize>>::value;
    info_for_overloading<char, func_fields>::type;
    join_overloading<std::decay_t<char>, func_fields>::value;

    auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
    auto aaa = visitor.visit('1');

    //has_member_func_fields<B<int, char>, visit_to_tuplize, field_visitor<pack_as_tuple, visit_to_tuplize>>::value;
    

    //B<long, short> b;
    //b.v1 = 1;
    //b.v2 = '1';
    std::array<double, 4> a = { 1, 2, 3, 4 };
    auto bbb = visitor.visit(a);
    //auto t = ;
    auto i = tuplize(std::make_pair(1, 2));
    std::cout << std::endl;

    static_assert(std::make_tuple(1, 2.0) == std::make_tuple(true, 2.0f), "");

    join_overloading<std::tuple<int, bool>, func_fields>::value;

    //auto bt = tuplize(b);
    //auto bt2 = b.tuplize();

    //std::vector<B<int, char>> bs(10, b);

}