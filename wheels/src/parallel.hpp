/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include <future>
#include <algorithm>
#include <numeric>
#include <vector>

#include "parallel_fwd.hpp"

namespace wheels {

// parallel_for_each
template <class FunT>
void parallel_for_each(size_t n, FunT &&fun, size_t batch_num,
                       size_t concurrency_num) {
  std::vector<std::thread> threads;
  threads.reserve(concurrency_num);
  for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
    size_t bfirst = bid * batch_num;
    size_t blast = std::min(n, (bid + 1) * batch_num) - 1;
    threads.emplace_back(
        [&fun](size_t first, size_t last) {
          for (size_t i = first; i <= last; i++) {
            fun(i);
          }
        },
        bfirst, blast);
    if (threads.size() >= concurrency_num || bid == n / batch_num) {
      for (auto &&t : threads) {
        t.join();
      }
      threads.clear();
    }
  }
}

template <class IterT, class FunT>
void parallel_for_each(IterT begin, IterT end, FunT &&fun, size_t batch_num,
                       size_t concurrency_num) {
  std::vector<std::thread> threads;
  threads.reserve(concurrency_num);

  auto n = std::distance(begin, end);
  for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
    size_t bfirst = bid * batch_num;
    size_t blast = std::min(n, (bid + 1) * batch_num) - 1;
    threads.emplace_back(
        [&fun](IterT first, IterT last) {
          while (first != last) {
            fun(*first);
            ++first;
          }
        },
        begin + bfirst, begin + blast);
    if (threads.size() >= concurrency_num || bid == n / batch_num) {
      for (auto &&t : threads) {
        t.join();
      }
      threads.clear();
    }
  }
}

// parallel_reduce
template <class IterT, class T, class ReduceT>
T parallel_reduce(IterT begin, IterT end, const T &initial, ReduceT&& redux,
                  size_t batch_num, size_t concurrency_num) {
  std::vector<std::future<T>> futures;
  futures.reserve(concurrency_num);

  T result = initial;
  auto n = std::distance(begin, end);
  for (size_t bid = 0; bid < n / batch_num + 1; bid++) {
    size_t bfirst = bid * batch_num;
    size_t blast = std::min<size_t>(n, (bid + 1) * batch_num) - 1;
    std::packaged_task<T(IterT, IterT)> task([&redux](IterT first, IterT last) {
      T tmp_result = 0;
      while (first != last) {
        tmp_result = redux(tmp_result, *first);
        ++first;
      }
      return tmp_result;
    });

    futures.push_back(task.get_future());

    std::thread(std::move(task), begin + bfirst, begin + blast).detach();
    if (futures.size() >= concurrency_num || bid == n / batch_num) {
      for (auto &&f : futures) {
        f.wait();
      }
      for (auto &&f : futures) {
        result = redux(result, f.get());
      }
      futures.clear();
    }
  }
  return result;
}
}
