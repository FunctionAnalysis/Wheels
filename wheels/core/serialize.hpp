#pragma once

#include <filesystem>
#include <fstream>

#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/forward_list.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/queue.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/stack.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

#include "fields.hpp"

namespace wheels {

// serializable
struct visit_to_serialize {};

template <class T> struct serializable {};

#define WHEELS_PARAMETER_DISTINGUISH(i) const_size<i> = const_size<i>()

// has_member_func_fields<const T &...>
template <class T, class ArcT, class = std::enable_if_t<has_member_func_fields<
                                   const T &, visit_to_serialize>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(0)) {
  static_cast<const T &>(data).fields(visit_to_serialize(), arc);
}
// has_member_func_fields<T & ...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              has_member_func_fields<T &, visit_to_serialize>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(1)) {
  const_cast<T &>(data).fields(visit_to_serialize(), arc);
}
// has_member_func_fields_simple<const T &...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              has_member_func_fields_simple<const T &>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(2)) {
  static_cast<const T &>(data).fields(arc);
}
// has_member_func_fields_simple<T &...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<const T &>::value &&
              has_member_func_fields_simple<T &>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(3)) {
  const_cast<T &>(static_cast<const T &>(data)).fields(arc);
}
// has_global_func_fields<const T & ...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<const T &>::value &&
              !has_member_func_fields_simple<T &>::value &&
              has_global_func_fields<const T &, visit_to_serialize>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(4)) {
  ::wheels::fields(static_cast<const T &>(data), visit_to_serialize(), arc);
}
// has_global_func_fields<T & ...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<const T &>::value &&
              !has_member_func_fields_simple<T &>::value &&
              !has_global_func_fields<const T &, visit_to_serialize>::value &&
              has_global_func_fields<T &, visit_to_serialize>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(5)) {
  ::wheels::fields(const_cast<T &>(static_cast<const T &>(data)),
                   visit_to_serialize(), arc);
}
// has_global_func_fields_simple<const T & ...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<const T &>::value &&
              !has_member_func_fields_simple<T &>::value &&
              !has_global_func_fields<const T &, visit_to_serialize>::value &&
              !has_global_func_fields<T &, visit_to_serialize>::value &&
              has_global_func_fields_simple<const T &>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(6)) {
  ::wheels::fields(static_cast<const T &>(data), arc);
}
// has_global_func_fields_simple<T & ...>
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<const T &, visit_to_serialize>::value &&
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<const T &>::value &&
              !has_member_func_fields_simple<T &>::value &&
              !has_global_func_fields<const T &, visit_to_serialize>::value &&
              !has_global_func_fields<T &, visit_to_serialize>::value &&
              !has_global_func_fields_simple<const T &>::value &&
              has_global_func_fields_simple<T &>::value>>
void save(ArcT &arc, const serializable<T> &data,
          WHEELS_PARAMETER_DISTINGUISH(7)) {
  ::wheels::fields(const_cast<T &>(static_cast<const T &>(data)), arc);
}

// has_member_func_fields
template <class T, class ArcT, class = std::enable_if_t<has_member_func_fields<
                                   T &, visit_to_serialize>::value>>
void load(ArcT &arc, serializable<T> &data, WHEELS_PARAMETER_DISTINGUISH(0)) {
  static_cast<T &>(data).fields(visit_to_serialize(), arc);
}
// has_member_func_fields_simple
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              has_member_func_fields_simple<T &>::value>>
void load(ArcT &arc, serializable<T> &data, WHEELS_PARAMETER_DISTINGUISH(1)) {
  static_cast<T &>(data).fields(arc);
}
// has_global_func_fields
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<T &>::value &&
              has_global_func_fields<T &, visit_to_serialize>::value>>
void load(ArcT &arc, serializable<T> &data, WHEELS_PARAMETER_DISTINGUISH(2)) {
  ::wheels::fields(static_cast<T &>(data), visit_to_serialize(), arc);
}
// has_global_func_fields_simple
template <class T, class ArcT,
          class = std::enable_if_t<
              !has_member_func_fields<T &, visit_to_serialize>::value &&
              !has_member_func_fields_simple<T &>::value &&
              !has_global_func_fields<T &, visit_to_serialize>::value &&
              has_global_func_fields_simple<T &>::value>>
void load(ArcT &arc, serializable<T> &data, WHEELS_PARAMETER_DISTINGUISH(3)) {
  ::wheels::fields(static_cast<T &>(data), arc);
}

#undef WHEELS_PARAMETER_DISTINGUISH

using namespace std::experimental;

// write
template <class ArchiveT = cereal::PortableBinaryOutputArchive, class... T>
inline bool write(const filesystem::path &filename, const T &... data) {
  std::ofstream out(filename, std::ios::binary);
  if (!out.is_open()) {
    println("file \"", filename, "\" cannot be saved!");
    return false;
  }
  try {
    ArchiveT archive(out);
    archive(data...);
    println("file \"", filename, "\" saved");
    out.close();
  } catch (std::exception &e) {
    println(e.what());
    return false;
  }
  return true;
}
// write_tmp
template <class ArchiveT = cereal::PortableBinaryOutputArchive, class... T>
inline bool write_tmp(const filesystem::path &filename, const T &... data) {
  return write<ArchiveT>(filesystem::temp_directory_path() / filename, data...);
}

// read
template <class ArchiveT = cereal::PortableBinaryInputArchive, class... T>
inline bool read(const filesystem::path &filename, T &... data) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    println("file \"", filename, "\" cannot be loaded!");
    return false;
  }
  try {
    ArchiveT archive(in);
    archive(data...);
    println("file \"", filename, "\" loaded");
    in.close();
  } catch (std::exception &e) {
    println(e.what());
    return false;
  }
  return true;
}

// read_tmp
template <class ArchiveT = cereal::PortableBinaryInputArchive, class... T>
inline bool read_tmp(const filesystem::path &filename, T &... data) {
  return read<ArchiveT>(filesystem::temp_directory_path() / filename, data...);
}
}