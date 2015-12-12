# Wheels
A C++ Toolkit for Geometry, Graphics and Vision

## Implemented Features
### generic tensors
```cpp
using namespace wheels;
// t1: a 3x4x5 double type tensor filled with 1's
auto t1 = ones(3, 4, 5).eval(); 
// t2: a 2x2x2x2 complex<double> type tensor filled with 0's
auto t2 = zeros<std::complex<double>>(2, 2, 2, 2).eval(); 
// t3: a 3-vector, with static shape, initialized with 1, 2, 3
vec3 t3(1, 2, 3);
// t4: a 3-vector, with dynamic shape, initialized with 1, 2, 3
vecx t4(1, 2, 3);
// t5: a 5x5 matrix, with dynamic shape, filled with 5's
matx t5(make_shape(5, 5), 5);
using namespace wheels::literals;
// t6: a 3x4x5 tensor, with static shape, filled with 123's
// here *_c literal operator is provided to define compile time integral constants
auto t6 = constants(make_shape(3_c, 4_c, 5_c), 123.0).eval();
```
### stack/heap allocation
```cpp
// static shaped tensor types are stack allocated, and satisfy standard layout
static_assert(sizeof(vec_<double, 3>) == 24, ""); 
static_assert(std::is_standard_layout<vec_<double, 3>>::value, "");
static_assert(sizeof(mat_<double, 2, 2>) == 32, ""); 
static_assert(std::is_standard_layout<mat_<double, 2, 2>>::value, "");
// dynamic shaped tensor types are heap allocated
static_assert(sizeof(vecx(1)) == 48, "");
static_assert(sizeof(vecx(1, 2)) == 48, "");
static_assert(sizeof(vecx(1, 2, 3, 4, 5, 6, 7, 8)) == 48, "");
```
### distinguish tensor types in compile time
```cpp
// deal with all kinds of tensors
template <class ShapeT, class EleT, class DerivedT>
void foo(const tensor_base<ShapeT, EleT, DerivedT> & t){
  // ...
}
// deal with matrices of complex numbers
template <class RowsT, class ColsT, class T, class DerivedT>
void foo(const tensor_base<tensor_shape<size_t, RowsT, ColsT>, std::complex<T>, DerivedT> & t){
  // ...
}
```
### delayed evaluation
```cpp
auto t1 = ones(100, 200);
auto r1 = sin(t1);
auto r2 = r1 + t1 * 2.0;
auto rr = min(t1, r2).t().eval(); // evaluated only when .eval() is called
```
### more...
```cpp
// element retreival
auto efirst = rr[0];     // via vectoized index
auto efirst2 = rr(0, 0); // via tensor subscripts
// index tags can be used to represent sizes
using namespace wheels::index_tags;
auto e1 = rr[length - 1]; // same with rr[100*200-1]
auto e2 = rr(length / 2, (length - 20) / 2);   // same with rr(100/2, (200-20)/2)
auto e3 = rr(10, (length / 10 + 2) * 2); // same with rr(10, (200/10+2)*2)
auto e4 = rr(last, last / 3);            // last = length-1

// symbolic expressions
auto fun = max(0_symbol + 1_symbol * 2);
auto result1 = fun(3, 2); // 0_symbol->3, 1_symbol->2, scalar calculation
auto result2 = fun(vec3(2, 3, 4), ones(3)); // 0_symbol->vec3(2, 3, 4), 1_symbol->ones(3), vector calculation

// wheels classes are all serializable using cereal
write(filesystem::temp_directory_path() / "vec3.cereal", vec3(1, 2, 3));
vec3 v;
read(filesystem::temp_directory_path() / "vec3.cereal", v); // v == vec3(1, 2, 3)
```
