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
    B() : v1(), v2() {}
    B(const T1 & a, const T2 & b) : v1(a), v2(b) {}
    template <class V>
    auto fields(V && v) & {
        return v(v1, v2);
    }
    template <class V>
    auto fields(V && v) const & {
        return v(v1, v2);
    }
    template <class V>
    auto fields(V && v) && {
        return v(std::move(v1), std::move(v2));
    }
    T1 v1;
    T2 v2;
};


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

struct C {
    B<char, int> b1;
    B<int, char> b2;
};

namespace wheels {
    template <class V>
    auto fields(const C & c, V && visitor) {
        return visitor(c.b1, c.b2);
    }
    template <class V>
    auto fields(C & c, V && visitor) {
        return visitor(c.b1, c.b2);
    }
}

TEST(core, fields2) {
    C c = { {'1', 1}, {1, '1'} };
    auto ct = tuplize(c);
    static_assert(type_of(ct).decay() == types<std::tuple<std::tuple<char &, int &>, std::tuple<int &, char &>>>(), "");
    B<int, int> b1 = { 0, 0 };
    B<int, int> b2 = { 0, 0 };
    std::forward_as_tuple(b1.as_tuple(), b2.as_tuple()) = tuplize(c);
    ASSERT_EQ(b1.v1, '1');
    ASSERT_EQ(b1.v2, 1);
    ASSERT_EQ(b2.v1, 1);
    ASSERT_EQ(b2.v2, '1');

    ASSERT_TRUE(b1 != b2);
}

struct D {
    std::vector<C> cs;
    B<int, long> b;
    template <class V>
    auto fields(V && v) & {
        return v(cs, b);
    }
    template <class V>
    auto fields(V && v) const & {
        return v(cs, b);
    }
};

TEST(core, fields3) {
    D d = { {
        {{'1', 1}, {1, '1'}}, 
        {{'2', 2}, {2, '2'}}, 
        {{'3', 3}, {3, '3'}}
        }, {10, 10} };
    auto dt = tuplize(d);
    std::vector<D> ds = { d, d };
    auto dts = tuplize(ds);
    decltype(auto) dt1 = dts[0];
    C c = { { '4', 4 }, { 4, '4' } };
    std::get<0>(dt1)[0] = tuplize(c);

    ASSERT_EQ(ds[0].cs[0].b1.v1, '4');

    std::list<D> ds2 = { ds.front(), d };
    ASSERT_TRUE(tuplize(ds) == tuplize(ds2));
    ASSERT_TRUE(tuplize(ds) >= tuplize(ds2));
    ASSERT_TRUE(tuplize(ds) <= tuplize(ds2));
    ASSERT_FALSE(tuplize(ds) > tuplize(ds2));
    ASSERT_FALSE(tuplize(ds) < tuplize(ds2));
}
