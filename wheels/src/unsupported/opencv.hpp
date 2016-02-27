#pragma once

#include "../core/macros.hpp"
#if defined(wheels_compiler_msc)
#pragma warning(push, 0)
#endif
#include <opencv2/opencv.hpp>
#if defined(wheels_compiler_msc)
#pragma warning(pop)
#endif

#include "../tensor/map.hpp"
#include "../tensor/tensor.hpp"

namespace cv {

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

namespace wheels {

template <class T, int M, int N, class U, class V>
decltype(auto) fields(const cv::Matx<T, M, N> &mat, U &&, V &&visitor) {
  return visitor(mat.val);
}
template <class T, int M, int N, class U, class V>
decltype(auto) fields(cv::Matx<T, M, N> &mat, U &&, V &&visitor) {
  return visitor(mat.val);
}

template <class T, class U, class V>
decltype(auto) fields(const cv::Mat_<T> &mat, U &&, V &&visitor) {
  return visitor(as_container(mat, forward<V>(visitor)));
}
template <class T, class U, class V>
decltype(auto) fields(cv::Mat_<T> &mat, U &&, V &&visitor) {
  return visitor(as_container(mat, forward<V>(visitor)));
}
template <class T, class U, class V>
decltype(auto) fields(cv::Mat_<T> &&mat, U &&, V &&visitor) {
  return visitor(as_container(std::move(mat), forward<V>(visitor)));
}

namespace details {
cv::Mat _imread(const std::string &path);
bool _imwrite(const std::string &path, const cv::Mat &mat);
}

// cv_image
template <class T = uint8_t, size_t Depth = 3>
class cv_image
    : public tensor_base<vec_<T, Depth>, tensor_shape<size_t, size_t, size_t>,
                         cv_image<T, Depth>> {
  static constexpr int _idetph = static_cast<int>(Depth);

public:
  cv_image() {}
  cv_image(const std::string &path) : mat(details::_imread(path)) {}
  cv_image(const cv::Mat_<cv::Vec<T, Depth>> &m) : mat(m) {}
  template <wheels_enable_if(Depth == 1)>
  cv_image(const cv::Mat_<T> &m) : mat(m) {}

  constexpr bool null() const { return mat.empty(); }

  bool write(const std::string &path) const {
    return details::_imwrite(path, mat);
  }

  operator cv::Mat_<cv::Vec<T, Depth>>() const { return mat; }
  template <wheels_enable_if(Depth == 1)> operator cv::Mat_<T>() const {
    return mat;
  }

  cv_image(const cv_image &t) { t.mat.copyTo(mat); }
  cv_image(cv_image &&t) { cv::swap(mat, t.mat); }
  cv_image &operator=(const cv_image &t) {
    t.mat.copyTo(mat);
    return *this;
  }
  cv_image &operator=(cv_image &&t) {
    cv::swap(mat, t.mat);
    return *this;
  }

  template <class AnotherT>
  constexpr cv_image(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherT>
  cv_image &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

public:
  template <class ArcT> void serialize(ArcT &ar) { ar(mat); }
  template <class V> constexpr decltype(auto) fields(V &&v) const {
    return v(mat);
  }
  template <class V> decltype(auto) fields(V &&v) { return v(mat); }

public:
  cv::Mat_<cv::Vec<T, _idetph>> mat;
};

// shape_of
template <class T, size_t Depth>
constexpr auto shape_of(const cv_image<T, Depth> &t) {
  return tensor_shape<size_t, size_t, size_t>(t.mat.rows, t.mat.cols);
}

// reserve_shape
template <class T, size_t Depth, class ST, class... SizeTs>
void reserve_shape(cv_image<T, Depth> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  static constexpr int _idetph = static_cast<int>(Depth);
  if (shape.magnitude() == numel(t)) {
    int newss[] = {(int)shape.at(const_index<0>()),
                   (int)shape.at(const_index<1>())};
    t.mat.reshape(_idetph, 2, newss);
  } else {
    t.mat = cv::Mat_<cv::Vec<T, _idetph>>((int)shape.at(const_index<0>()),
                                          (int)shape.at(const_index<1>()));
  }
}

// element_at
template <class T, size_t Depth, class SubT1, class SubT2>
constexpr tensor_map<const T, tensor_shape<size_t, const_size<Depth>>,
                     const T *>
element_at(const cv_image<T, Depth> &t, const SubT1 &s1, const SubT2 &s2) {
  return map(make_shape(const_size<Depth>()),
             t.mat(static_cast<int>(s1), static_cast<int>(s2)).val);
}

template <class T, size_t Depth, class SubT1, class SubT2>
inline tensor_map<T, tensor_shape<size_t, const_size<Depth>>, T *>
element_at(cv_image<T, Depth> &t, const SubT1 &s1, const SubT2 &s2) {
  return map(make_shape(const_size<Depth>()),
             t.mat(static_cast<int>(s1), static_cast<int>(s2)).val);
}

// for_each_element
template <class FunT, class T, size_t Depth, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT &&fun,
                      const cv_image<T, Depth> &t, Ts &&... ts) {
  static constexpr int _idetph = static_cast<int>(Depth);
  t.mat.forEach([&](const cv::Vec<T, _idetph> &e, const int *position) {
    fun(::wheels::map(make_shape(const_size<Depth>()), e.val),
        element_at(ts, position[0], position[1])...);
  });
  return true;
}
template <class FunT, class T, size_t Depth, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT &&fun,
                      cv_image<T, Depth> &t, Ts &&... ts) {
  static constexpr int _idetph = static_cast<int>(Depth);
  t.mat.forEach([&](cv::Vec<T, _idetph> &e, const int *position) {
    fun(::wheels::map(make_shape(const_size<Depth>()), e.val),
        element_at(ts, position[0], position[1])...);
  });
  return true;
}

// imread
template <class T = uint8_t, size_t Depth = 3>
inline auto imread(const std::string &path) {
  return cv_image<T, Depth>(path);
}
// imread
template <class T = uint8_t, size_t Depth = 3>
inline bool imwrite(const std::string &path,
                    const cv_image<T, Depth> &im) {
  return im.write(path);
}

//// cv_video_props
//class cv_video_props : public serializable<cv_video_props> {
//public:
//  cv_video_props() : fps(0), fourCC(0) {}
//
//public:
//  template <class V> constexpr decltype(auto) fields(V &&visitor) const {
//    return visitor(fps, fourCC);
//  }
//  template <class V> decltype(auto) fields(V &&visitor) {
//    return visitor(fps, fourCC);
//  }
//
//public:
//  double fps;
//  int fourCC;
//};

//namespace details {
//std::vector<cv::Mat> _vdread(const std::string &path,
//                             cv_video_props *vp = nullptr);
//size_t _vdread(const std::string &path,
//               const std::function<bool(const cv::Mat &fram)> &processor,
//               cv_video_props *vp = nullptr);
//bool _vdwrite(const std::string &path, const std::vector<cv::Mat> &frames,
//              const cv_video_props &props);
//}

//// cv_video
// template <class T, size_t Depth>
// class cv_video
//    : public tensor_base<
//          T, tensor_shape<size_t, size_t, size_t, size_t, const_size<Depth>>,
//          cv_video<T, Depth>> {
//  static constexpr int _idetph = static_cast<int>(Depth);
//
// public:
//  using value_type = T;
//  using shape_type =
//      tensor_shape<size_t, size_t, size_t, size_t, const_size<Depth>>;
//  cv_video() {}
//  explicit cv_video(const std::string &path) {
//    auto frms = details::_vdread(path, &props);
//    frames.reserve(frms.size());
//    for (auto &f : frms) {
//      frames.emplace_back(f);
//    }
//  }
//
//  bool write(const std::string &path) const {
//    std::vector<cv::Mat> frms;
//    frms.reserve(frames.size());
//    for (auto &f : frames) {
//      frms.push_back(f.mat);
//    }
//    return details::_vdwrite(path, frms, props);
//  }
//
//  cv_video(const cv_video &) = default;
//  cv_video(cv_video &&) = default;
//  cv_video &operator=(const cv_video &) = default;
//  cv_video &operator=(cv_video &&) = default;
//
//  template <class AnotherT>
//  constexpr cv_video(const tensor_core<AnotherT> &another) {
//    assign_elements(*this, another.derived());
//  }
//  template <class AnotherT>
//  cv_video &operator=(const tensor_core<AnotherT> &another) {
//    assign_elements(*this, another.derived());
//    return *this;
//  }
//
// public:
//  template <class SubT1, class SubT2, class SubT3>
//  decltype(auto) operator()(const SubT1 &fram, const SubT2 &r,
//                            const SubT3 &c) const {
//    return frames[fram](r, c);
//  }
//  template <class SubT1, class SubT2, class SubT3>
//  decltype(auto) operator()(const SubT1 &fram, const SubT2 &r, const SubT3 &c)
//  {
//    return frames[fram](r, c);
//  }
//
//  template <class SubT1, class SubT2, class SubT3, class SubT4>
//  decltype(auto) operator()(const SubT1 &fram, const SubT2 &r, const SubT3 &c,
//                            const SubT4 &channel) const {
//    return frames[fram](r, c, channel);
//  }
//  template <class SubT1, class SubT2, class SubT3, class SubT4>
//  decltype(auto) operator()(const SubT1 &fram, const SubT2 &r, const SubT3 &c,
//                            const SubT4 &channel) {
//    return frames[fram](r, c, channel);
//  }
//
// public:
//  template <class ArcT> void serialize(ArcT &ar) { ar(props, frames); }
//  template <class V> constexpr decltype(auto) fields(V &&v) const {
//    return v(props, frames);
//  }
//  template <class V> decltype(auto) fields(V &&v) { return v(props, frames); }
//
// public:
//  cv_video_props props;
//  std::vector<cv_image<T, Depth>> frames;
//};
//
//// shape_of
// template <class T, size_t Depth> auto shape_of(const cv_video<T, Depth> &t) {
//  size_t rows = t.frames.empty() ? 0 : t.frames.front().rows;
//  size_t cols = t.frames.empty() ? 0 : t.frames.front().cols;
//  return tensor_shape<size_t, size_t, size_t, size_t, const_size<Depth>>(
//      t.frames.size(), rows, cols, std::ignore);
//}
//
//// reserve_shape
// template <class T, size_t Depth, class ST, class FramsT, class RowsT,
//          class ColsT, class DepthT>
// void reserve_shape(
//    cv_video<T, Depth> &t,
//    const tensor_shape<ST, FramsT, RowsT, ColsT, DepthT> &shape) {
//  static constexpr int _idetph = static_cast<int>(Depth);
//  assert(Depth == shape.at(const_index<3>()));
//  size_t newframs = shape.at(const_index<0>());
//  size_t newrows = shape.at(const_index<1>());
//  size_t newcols = shape.at(const_index<2>());
//  size_t oldrows = shape_of(t).at(const_index<1>());
//  size_t oldcols = shape_of(t).at(const_index<2>());
//  t.frames.resize(newframs);
//  const auto fram_shape = make_shape(newrows, newcols, const_size<Depth>());
//  for (cv_image<T, Depth> &im : t.frames) {
//    reserve_shape(im, fram_shape);
//  }
//}
//
//// element_at
// template <class T, size_t Depth, class SubT1, class SubT2, class SubT3,
//          class SubT4>
// constexpr decltype(auto) element_at(const cv_video<T, Depth> &t,
//                                    const SubT1 &fram, const SubT2 &row,
//                                    const SubT3 &col, const SubT4 &depth) {
//  return t(static_cast<size_t>(fram), static_cast<int>(row),
//           static_cast<int>(col), static_cast<int>(depth));
//}
// template <class T, size_t Depth, class SubT1, class SubT2, class SubT3,
//          class SubT4>
// decltype(auto) element_at(cv_video<T, Depth> &t, const SubT1 &fram,
//                          const SubT2 &row, const SubT3 &col,
//                          const SubT4 &depth) {
//  return t(static_cast<size_t>(fram), static_cast<int>(row),
//           static_cast<int>(col), static_cast<int>(depth));
//}
//
//// for_each_element
// template <class FunT, class T, size_t Depth, class... Ts>
// bool for_each_element(behavior_flag<unordered>, FunT &&fun,
//                      const cv_video<T, Depth> &t, Ts &&... ts) {
//  static constexpr int _idetph = static_cast<int>(Depth);
//  for (size_t f = 0; f < t.frames.size(); f++) {
//    t.frames[f].mat.forEach([&](cv::Vec<T, _idetph> &e, const int *position) {
//      for (size_t d = 0; d < Depth; d++) {
//        fun(e(d), element_at(ts, f, position[0], position[1], d)...);
//      }
//    });
//  }
//  return true;
//}
// template <class FunT, class T, size_t Depth, class... Ts>
// bool for_each_element(behavior_flag<unordered>, FunT &&fun,
//                      cv_video<T, Depth> &t, Ts &&... ts) {
//  static constexpr int _idetph = static_cast<int>(Depth);
//  for (size_t f = 0; f < t.frames.size(); f++) {
//    t.frames[f].mat.forEach([&](cv::Vec<T, _idetph> &e, const int *position) {
//      for (size_t d = 0; d < Depth; d++) {
//        fun(e(d), element_at(ts, f, position[0], position[1], d)...);
//      }
//    });
//  }
//  return true;
//}
}