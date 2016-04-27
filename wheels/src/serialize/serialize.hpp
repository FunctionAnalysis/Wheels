#pragma once

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

#include "../core/macros.hpp"
#include "../core/fields.hpp"

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

// write
template <class ArchiveT = cereal::PortableBinaryOutputArchive, class... T>
inline bool write(const std::string &filename, const T &... data) {
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

// read
template <class ArchiveT = cereal::PortableBinaryInputArchive, class... T>
inline bool read(const std::string &filename, T &... data) {
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
}

#if defined(wheels_compiler_msc)
#include <filesystem>
namespace wheels {
using namespace std::experimental;

// write_tmp
template <class ArchiveT = cereal::PortableBinaryOutputArchive, class... T>
inline bool write_tmp(const filesystem::path &filename, const T &... data) {
  return write<ArchiveT>(
      (filesystem::temp_directory_path() / filename).string(), data...);
}

// read_tmp
template <class ArchiveT = cereal::PortableBinaryInputArchive, class... T>
inline bool read_tmp(const filesystem::path &filename, T &... data) {
  return read<ArchiveT>((filesystem::temp_directory_path() / filename).string(),
                        data...);
}


// serialize storage
template <class T, class ShapeT, bool S> struct storage;
template <class T, class ShapeT, bool S, class ArcT>
void save(ArcT &ar, const storage<T, ShapeT, S> &st) {
  details::_save_shape(ar, st.shape());
  ar(cereal::binary_data(st.data(), sizeof(T) * st.shape().magnitude()));
}
template <class T, class ShapeT, bool S, class ArcT>
void load(ArcT &ar, storage<T, ShapeT, S> &st) {
  ShapeT s;
  details::_load_shape(ar, s);
  st.reshape(s);
  ar(cereal::binary_data(st.data(), sizeof(T) * s.magnitude()));
}
}


// serialize opencv
namespace cv {
class Mat;
template <class T, int M, int N> class Matx;
template <class T> struct Point_;
template <class T> struct Size_;
template <class T> struct Rect_;
struct KeyPoint;
struct Moments;

// MUST be defined in the namespace of the underlying type (cv::XXX),
//    definition of alias names in namespace pano::core won't work!
// see
// http://stackoverflow.com/questions/13192947/argument-dependent-name-lookup-and-typedef

// Serialization for cv::Mat
template <class Archive> void save(Archive &ar, Mat const &im) {
  // cv::solve(im);
  ar(im.elemSize(), im.type(), im.cols, im.rows);
  ar(cereal::binary_data(im.data, im.cols * im.rows * im.elemSize()));
}

// Serialization for cv::Mat
template <class Archive> void load(Archive &ar, Mat &im) {
  size_t elemSize;
  int type, cols, rows;
  ar(elemSize, type, cols, rows);
  im.create(rows, cols, type);
  ar(cereal::binary_data(im.data, cols * rows * elemSize));
}

// Serialization for cv::Matx<T, M, N>
template <class Archive, class T, int M, int N>
inline void serialize(Archive &ar, Matx<T, M, N> &m) {
  ar(m.val);
}

// Serialization for cv::Point_<T>
template <class Archive, class T>
inline void serialize(Archive &ar, Point_<T> &p) {
  ar(p.x, p.y);
}

// Serialization for cv::Size_<T>
template <class Archive, class T>
inline void serialize(Archive &ar, Size_<T> &s) {
  ar(s.width, s.height);
}

// Serialization for cv::Rect_<T>
template <class Archive, class T>
inline void serialize(Archive &ar, Rect_<T> &r) {
  ar(r.x, r.y, r.width, r.height);
}

// Serialization for cv::KeyPoint
template <class Archive> inline void serialize(Archive &ar, KeyPoint &p) {
  ar(p.pt, p.size, p.angle, p.response, p.octave, p.class_id);
}

// Serialization for cv::Moments
template <class Archive> inline void serialize(Archive &ar, Moments &m) {
  ar(m.m00, m.m10, m.m01, m.m20, m.m11, m.m02, m.m30, m.m21, m.m12, m.m03);
  ar(m.mu20, m.mu11, m.mu02, m.mu30, m.mu21, m.mu12, m.mu03);
  ar(m.nu20, m.nu11, m.nu02, m.nu30, m.nu21, m.nu12, m.nu03);
}
}


#endif
