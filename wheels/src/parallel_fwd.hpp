#pragma once

#include <thread>

namespace wheels {

// parallel_for_each
template <class FunT>
void parallel_for_each(
    size_t n, FunT &&fun, size_t batch_num = 1,
    size_t concurrency_num = std::thread::hardware_concurrency());

// parallel_for_each
template <class IterT, class FunT>
void parallel_for_each(
    IterT begin, IterT end, FunT &&fun, size_t batch_num = 1,
    size_t concurrency_num = std::thread::hardware_concurrency());

// parallel_reduce
template <class IterT, class T, class ReduceT>
T parallel_reduce(IterT begin, IterT end, const T &initial, ReduceT &&redux,
                  size_t batch_num = 1,
                  size_t concurrency_num = std::thread::hardware_concurrency());

}
