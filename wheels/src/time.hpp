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

#include <chrono>
#include <iostream>

#include "utility.hpp"

namespace wheels {

// time_cost
template <class DurationT = std::chrono::milliseconds, class FunT>
auto time_cost(FunT &&fun) {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::forward<FunT>(fun)();
  auto d = std::chrono::high_resolution_clock::now() - start_time;
  return std::chrono::duration_cast<DurationT>(d);
}

namespace details {
inline const char *_period_str(const std::nano &) { return "nanoseconds"; }
inline const char *_period_str(const std::micro &) { return "microseconds"; }
inline const char *_period_str(const std::milli &) { return "milliseconds"; }
inline const char *_period_str(const std::ratio<1> &) { return "seconds"; }
inline const char *_period_str(const std::ratio<60> &) { return "minutes"; }
inline const char *_period_str(const std::ratio<3600> &) { return "hours"; }
template <intmax_t Nx, intmax_t Dx>
inline std::string _period_str(const std::ratio<Nx, Dx> &) {
  return "(x" + std::to_string(Nx) + "/" + std::to_string(Dx) + ") seconds";
}
}

template <class RepT, class PeriodT>
std::ostream &operator<<(std::ostream &os,
                         const std::chrono::duration<RepT, PeriodT> &d) {
  return print_to(os, d.count(), " ", details::_period_str(PeriodT()));
}
}
