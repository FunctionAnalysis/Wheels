#pragma once

#include <chrono>
#include <iostream>

#include "utility.hpp"

namespace wheels {

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
