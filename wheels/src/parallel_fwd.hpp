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
