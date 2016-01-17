# Wheels
A C++ Toolkit for Tensor Manipulations.

## Features
### generic tensors
Wheels provide tensor types of varies ranks, shapes and element types, from small fix-sized vectors allocated on stack to large dynamic-sized matrices allocated on heap.
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
// t5: a 500x500 matrix, with dynamic shape, filled with 5's
matx t5(make_shape(500, 500), 5);
using namespace wheels::literals;
// t6: a 3x4x5 tensor, with static shape, filled with 123's
// here *_c literal operator is provided to define compile time integral constants
auto t6 = constants(make_shape(3_c, 4_c, 5_c), 123.0).eval();
```
### lazy evaluation
Lazy evaluation is used to improve computation efficiency and to save memory.
```cpp
auto t1 = ones(100, 200);
auto r1 = sin(t1);
auto r2 = r1 + t1 * 2.0;
auto result = min(t1, r2).t().eval(); // evaluated only when .eval() is called
```
### static polymorphism
Static polymorphism is employed to unify the behavior of all tensor types, including the interim result types in lazy evaluation. 
We can implement different functions for different tensors without considering their underlying types.
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
### more tools ...
Index tags can be used in element retrieval.
```cpp
using namespace wheels::index_tags;
auto t = ones(100, 200).eval(); // a 100x200 matrix
auto efirst = t[10];      // element at vectoized index 10
auto efirst2 = t(20, 30); // element at tensor subscripts (20, 30)
auto e1 = t[length - 1];  // same with t[100*200-1]
auto e2 = t(length / 2, (length - 20) / 2);   // same with t(100/2, (200-20)/2)
auto e3 = t(10, (length / 10 + 2) * 2);       // same with t(10, (200/10+2)*2)
auto e4 = t(last, last / 3);                  // last = length-1
```
Symbolic expressions. 
```cpp
auto fun = max(0_symbol + 1, 1_symbol * 2);
auto result1 = fun(3, 2);                          // 0_symbol->3, 1_symbol->2, result1 = 4 of int
auto result2 = fun(vec3(2, 3, 4), ones(3)).eval(); // 0_symbol->vec3(2, 3, 4), 1_symbol->ones(3), result2 = [4, 4, 5] of vec3
```
Tensors are all serializable based on [cereal](https://github.com/USCiLab/cereal).
```cpp
write(filesystem::temp_directory_path() / "vec3.cereal", vec3(1, 2, 3));
vec3 v;
read(filesystem::temp_directory_path() / "vec3.cereal", v); // v == vec3(1, 2, 3)
```

## Tested Compilers
Visual Studio 2015
