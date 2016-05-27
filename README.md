# Wheels
Tensors for C++ programming.

## Tested Compilers
* Visual Studio 2015
* MinGW GCC 5.3.0

## Examples
Let's start with the standard hello world program:
```cpp
#include <wheels/tensor>

using namespace wheels;
using namespace wheels::literals; // to use user defined literals like '_ts'
using namespace wheels::tags;     // to use index tags like 'length', 'last' ...

int main(){
	auto greeting = "hello world!"_ts;
	println(greeting);
}
```
More with the string:
```cpp
auto greeting = "hello world! 123456"_ts;
println(greeting);

// show only letters
println(greeting[where('a' <= greeting && greeting <= 'z' ||
                       'A' <= greeting && greeting <= 'Z')]);

// print the reversed string without data copy
println(greeting[last - iota(length)]);

// concatenate the strings without data copy
println(cat(greeting, " "_ts, "let's rock!"_ts));

// promote the string from a vector to a matrix,
// repeat it along rows, and transpose it.
// all without data copy!
println(repeat(promote(1_c, greeting), 3, 1).t());
```

## Features
### Generic Tensor Types
Wheels provide tensor types of varies ranks, shapes and element types, from small fix-sized vectors allocated on stack to large dynamic-sized multi-dimensional arrays allocated on heap.
```cpp
using namespace wheels;

// t1: a 3x4x5 double type tensor filled with 1's
auto t1 = ones(3, 4, 5); 

// t2: a 2x2x2x2 complex<double> type tensor filled with 0's
auto t2 = zeros<std::complex<double>>(2, 2, 2, 2);

// t3: a 3-vector, with static shape, initialized with 1, 2, 3
vec3 t3(1, 2, 3);

// t4: a 3-vector, with dynamic shape, initialized with 1, 2, 3
vecx t4(1, 2, 3);

// t5: a 500x500 matrix, with dynamic shape, filled with 5's
matx t5(make_shape(500, 500), 5);

using namespace wheels::literals;

// t6: a 3x4x5 tensor, with static shape, filled with 123's
// here *_c literal operator is provided to define 
// compile time integral constants
auto t6 = constants(make_shape(3_c, 4_c, 5_c), 123.0).eval();
```
### Lazy Evaluation
Lazy evaluation is used to improve computation efficiency and to save memory.
```cpp
auto t1 = ones(100, 200);
auto r1 = sin(t1);
auto r2 = r1 + t1 * 2.0;
auto result = min(t1, r2).t();

// all computations are evaluated only when .eval() is called
auto eval_result = result.eval(); 
```
### Static Polymorphism
Static polymorphism is employed to unify the behavior of all tensor types, including the interim result types in lazy evaluation. 
Therefore we can implement different functions for different tensors without considering their underlying types.
```cpp
// deal with all kinds of tensors
template <class EleT, class ShapeT, class DerivedT>
void foo(const tensor_base<EleT, ShapeT, DerivedT> & t){
  // ...
}
// only deal with a rank-3 tensor containing complex numbers
template <class T, class S1, class S2, class S3, class DerivedT>
void foo(const tensor_base<std::complex<T>, 
                           tensor_shape<size_t, S1, S2, S3>,
                           DerivedT> &t) {
  // ...
}
```
### Index Tags
Index tags can make elements retrieval more convenient.
```cpp
using namespace wheels::tags;
auto t = ones(100, 200).eval(); // a 100x200 matrix
auto efirst = t[10];      // element at vectoized index 10
auto efirst2 = t(20, 30); // element at tensor subscripts (20, 30)
auto e1 = t[length - 1];  // same with t[100*200-1]
auto e2 = t(length / 2, (length - 20) / 2);   // same with t(100/2, (200-20)/2)
auto e3 = t(10, (length / 10 + 2) * 2);       // same with t(10, (200/10+2)*2)
auto e4 = t(last, last / 3);                  // last = length-1
auto e5 = t(iota(length / 2) * 2, last);      // same with t(iota(100/2)*2, 199)
```

### What is More
* Compile time constant integers
* Compile time symbolic expressions
* ...
