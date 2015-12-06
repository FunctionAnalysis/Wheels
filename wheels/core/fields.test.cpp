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

template <class T1, class T2>
struct B : comparable<B<T1, T2>> {
    B(const T1 & a, const T2 & b) : v1(a), v2(b) {}
    template <class U, class V>
    auto fields(U &&, V && v) & {
        return v(v1, v2);
    }
    template <class U, class V>
    auto fields(U &&, V && v) const & {
        return v(v1, v2);
    }
    template <class U, class V>
    auto fields(U &&, V && v) && {
        return v(std::move(v1), std::move(v2));
    }
    T1 v1;
    T2 v2;
};

TEST(core, fields_type_traits) {

    static_assert(join_overloading<char, func_fields>::value, "");

    static_assert(has_global_func_fields<char, visit_to_tuplize, field_visitor<pack_as_tuple, visit_to_tuplize>>::value, "");
    static_assert(has_global_func_fields<unsigned char, visit_to_tuplize, field_visitor<pack_as_tuple, visit_to_tuplize>>::value, "");

    static_assert(!has_global_func_fields<B<int, long>, visit_to_tuplize, field_visitor<pack_as_tuple, visit_to_tuplize>>::value, "");
    static_assert(has_member_func_fields<B<int, long>, visit_to_tuplize, field_visitor<pack_as_tuple, visit_to_tuplize>>::value, "");

}

TEST(core, fields) {
    auto bt = tuplize(B<char, int>('1', 1));
    static_assert(type_of(bt).decay() == types<std::tuple<char, int>>(), "");    
    ASSERT_TRUE(bt == std::make_tuple('1', 1));

    B<int, long long> b2(2, 2ll);
    auto bt2 = tuplize(b2);
    static_assert(type_of(bt2).decay() == types<std::tuple<int &, long long &>>(), "");
    ASSERT_TRUE(bt2 == std::make_tuple(2, 2ll));
    ASSERT_TRUE((B<char, int>('2' - '0', 2) == b2));

    bt2 = std::make_tuple(3, 3);
    ASSERT_EQ(b2.v1, 3);
    ASSERT_EQ(b2.v2, 3);
}