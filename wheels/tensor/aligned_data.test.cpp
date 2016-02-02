#include <gtest/gtest.h>

#include "aligned_data.hpp"
#include "tensor.hpp"

using namespace wheels;

struct A {};
struct B : A {};
struct C : B {};
struct D : C {};
struct E : D {};

void foo(const A &) {std::cout << "A" << std::endl;}
void foo(const C &) {std::cout << "C" << std::endl;}

TEST(tensor, test) {
    E d;
    foo(d);
}

TEST(tensor, aligned_data) {

    vec3 v1;
    v1[0];
    vec_<bool, 5> v2;
    v2.ptr();

    element_at_index(v1, 0);

    for_each_element(order_flag<index_ascending>(),
                     [](auto e) { std::cout << e << ' '; }, v1);
}