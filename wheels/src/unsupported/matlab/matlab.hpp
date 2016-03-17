#pragma once

#include <complex>
#include <vector>

#include <mat.h>
#include <mex.h>

#include "../../core/types.hpp"
#include "../../core/const_expr.hpp"
#include "../../core/const_ints.hpp"

#include "../../tensor/tensor.hpp"

namespace wheels {

namespace details {

template <class T> struct mx_type_to_class_id {
  static const mxClassID value = mxUNKNOWN_CLASS;
};
template <mxClassID ClassID> struct mx_class_id_to_type { using type = void; };

#define WHEELS_BIND_TYPE_WITH_MX_CLASSID(T, id)                                \
  template <> struct mx_type_to_class_id<T> {                                  \
    static const mxClassID value = id;                                         \
  };                                                                           \
  template <> struct mx_class_id_to_type<id> { using type = T; };

WHEELS_BIND_TYPE_WITH_MX_CLASSID(bool, mxLOGICAL_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(double, mxDOUBLE_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(float, mxSINGLE_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(int8_t, mxINT8_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(int16_t, mxINT16_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(int32_t, mxINT32_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(int64_t, mxINT64_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(uint8_t, mxUINT8_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(uint16_t, mxUINT16_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(uint32_t, mxUINT32_CLASS)
WHEELS_BIND_TYPE_WITH_MX_CLASSID(uint64_t, mxUINT64_CLASS)
#undef WHEELS_BIND_TYPE_WITH_MX_CLASSID

template <class ET, class ShapeT, class T, size_t... Is>
mxArray *_make_mx_array_seq(const tensor_base<ET, ShapeT, T> &ts,
                            const no &cplx, const const_ints<size_t, Is...> &) {
  static_assert(!is_complex<ET>::value, "invalid type");
  static const mxClassId classID = mx_type_to_class_id<ET>::value;
  static_assert(classID != mxUNKNOWN_CLASS, "type not supported in MATLAB");
  static const size_t rank = ShapeT::rank;
  const size_t dims[] = {ts.size(const_index<Is>())...};
  mxArray *mx = mxCreateNumericArray(rank, dims, classID, mxREAL);

  ET *data = static_cast<ET *>(mxGetData(mx));
  for_each_subscript(ts.shape(), [&data, &ts, &mx](auto &&... subs) {
    size_t subs_data[] = {static_cast<size_t>(subs)...};
    size_t offset = mxCalcSingleSubscript(mx, rank, subs_data);
    data[offset] = ts(subs...);
  });
  return mx;
}

template <class ET, class ShapeT, class T, size_t... Is>
mxArray *_make_mx_array_seq(const tensor_base<ET, ShapeT, T> &ts,
                            const yes &cplx,
                            const const_ints<size_t, Is...> &) {
  static_assert(is_complex<ET>::value, "invalid type");
  using real_value_t = typename real_component<ET>::type;
  static const mxClassId classID = mx_type_to_class_id<real_value_t>::value;
  static_assert(classID != mxUNKNOWN_CLASS, "type not supported in MATLAB");

  static const size_t rank = ShapeT::rank;
  const size_t dims[] = {ts.size(const_index<Is>())...};
  mxArray *mx = mxCreateNumericArray(rank, dims, classID, mxCOMPLEX);

  ET *real_data = static_cast<ET *>(mxGetData(mx));
  ET *imag_data = static_cast<ET *>(mxGetImagData(mx));
  for_each_subscript(
      ts.shape(), [&real_data, &imag_data, &ts, &mx](auto &&... subs) {
        size_t subs_data[] = {static_cast<size_t>(subs)...};
        size_t offset = mxCalcSingleSubscript(mx, rank, subs_data);
        auto e = ts(subs...);
        real_data[offset] = e.real();
        imag_data[offset] = e.imag();
      });
  return mx;
}
}

// make_mx_array
template <class ET, class ShapeT, class T>
inline mxArray *make_mx_array(const tensor_base<ET, ShapeT, T> &ts) {
  return details::_make_mx_array_seq(
      ts, details::is_complex<ET>(),
      make_rank_sequence(ts.shape()));
}

class matlab_mxarray {
public:
  matlab_mxarray();
  matlab_mxarray(mxArray *m, bool dos = false);
  matlab_mxarray(const matlab_mxarray &mx);
  matlab_mxarray(matlab_mxarray &&mx);
  ~matlab_mxarray();

  matlab_mxarray &operator=(const matlab_mxarray &mx);
  matlab_mxarray &operator=(matlab_mxarray &&mx);

  template <class ET, class ShapeT, class T>
  matlab_mxarray(const tensor_base<ET, ShapeT, T> &ts, bool dos = false)
      : _m(make_mx_array(ts)), _destroy_when_out_of_scope(dos) {}

public:
  mxArray *mxa() const;
  bool null() const { return _m == nullptr; }

  bool is_numeric() const;
  bool is_cell() const;
  bool is_logical() const;
  bool is_char() const;
  bool is_struct() const;
  bool is_opaque() const;
  bool is_function_handle() const;
  bool is_object() const;
  bool is_complex() const;
  bool is_sparse() const;
  bool is_double() const;
  bool is_single() const;
  bool is_int8() const;
  bool is_uint8() const;
  bool is_int16() const;
  bool is_uint16() const;
  bool is_int32() const;
  bool is_uint32() const;
  bool is_int64() const;
  bool is_uint64() const;

  template <class T> bool is() const { return false; }
  template <> bool is<bool>() const { return is_logical(); }
  template <> bool is<double>() const { return is_double(); }
  template <> bool is<float>() const { return is_single(); }
  template <> bool is<int8_t>() const { return is_int8(); }
  template <> bool is<uint8_t>() const { return is_uint8(); }
  template <> bool is<int16_t>() const { return is_int16(); }
  template <> bool is<uint16_t>() const { return is_uint16(); }
  template <> bool is<int32_t>() const { return is_int32(); }
  template <> bool is<uint32_t>() const { return is_uint32(); }
  template <> bool is<int64_t>() const { return is_int64(); }
  template <> bool is<uint64_t>() const { return is_uint64(); }

  size_t m() const;
  size_t n() const;
  void set_m(size_t m);
  void set_n(size_t n);

  bool empty() const;
  bool is_from_global_workspace() const;
  void set_is_from_global_workspace(bool b);

  size_t numel() const;
  std::string to_string() const;

  // as cell array
  matlab_mxarray cell(size_t i) const;
  void set_cell(size_t i, const matlab_mxarray &a);

  // as struct
  int nfields() const;
  const char *field_name(int n) const;
  int field_number(const std::string &name) const;
  matlab_mxarray field(const std::string &name, int i = 0) const;
  void set_field(const std::string &name, const matlab_mxarray &a,
                 int i = 0) const;

  matlab_mxarray property(const std::string &name, size_t i) const;
  void set_property(const std::string &name, int i, const matlab_mxarray &a);

  const char *class_name() const;

  size_t ndims() const;
  size_t rank() const { return ndims(); }
  std::vector<size_t> dims() const;
  size_t dim(size_t d) const;
  size_t size(size_t d) const { return dim(d); }
  size_t length() const;

  size_t single_sub(size_t dim, size_t *subs) const;
  template <class... Ints> size_t offset(Ints... subs) const {
    size_t sbs[] = {subs...};
    return single_sub(sizeof...(Ints), sbs);
  }

  void *real_data() const;
  void *imag_data() const;

  template <class T = double, class... Ints>
  decltype(auto) at_subs(Ints... subs) const {
    return _at_subs<T>(details::is_complex<T>(), subs...);
  }
  template <class T, class... Ints> void set_at_subs(const T &v, Ints... subs) {
    assert(is<T>());
    static_cast<T *>(real_data())[offset(subs...)] = v;
  }
  template <class T, class... Ints>
  void set_at_subs(const std::complex<T> &v, Ints... subs) {
    assert(is<T>() && is_complex());
    static_cast<T *>(real_data())[offset(subs...)] = v.real();
    static_cast<T *>(imag_data())[offset(subs...)] = v.imag();
  }

private:
  template <class T, class... Ints>
  const T &_at_subs(const no &is_cplx, Ints... subs) const {
    assert(is<T>());
    static_assert(std::is_arithmetic<T>::value, "invalid type");
    return static_cast<const T *>(real_data())[offset(subs...)];
  }
  template <class T, class... Ints>
  T _at_subs(const yes &is_cplx, Ints... subs) const {
    assert(is_complex());
    using real_t = typename details::real_component<T>::type;
    assert(is<real_t>());
    static_assert(std::is_arithmetic<real_t>::value, "invalid type");
    return T(static_cast<const real_t *>(real_data())[offset(subs...)],
             static_cast<const real_t *>(imag_data())[offset(subs...)]);
  }

private:
  mxArray *_m;
  bool _destroy_when_out_of_scope;
};


// matlab .mat file
class matlab_matfile {
public:
  enum OpeningMode {
    Read,
    Update,
    Write,
    Write_4,
    Write_CompressedData,
    Write_7_3
  };

public:
  matlab_matfile();
  explicit matlab_matfile(const std::string &fname, const std::string &mode);
  explicit matlab_matfile(const std::string &fname, OpeningMode mode);

  matlab_matfile(matlab_matfile &&a);
  matlab_matfile &operator=(matlab_matfile &&a);

  matlab_matfile(const matlab_matfile &) = delete;
  matlab_matfile &operator=(const matlab_matfile &a) = delete;

  virtual ~matlab_matfile();

public:
  bool null() const { return _fp == 0; }

  std::vector<std::string> var_names() const;

  matlab_mxarray var(const std::string &name) const;
  bool set_var(const std::string &name, const matlab_mxarray &mxa,
               bool as_global = false);
  bool remove_var(const std::string &name);

private:
  std::string _fname;
  void *_fp;
};

// the matlab engine
class matlab_engine {
public:
  matlab_engine(const std::string &defaultDir = std::string(),
                bool singleUse = false);
  ~matlab_engine();

  matlab_engine(matlab_engine &&e);
  matlab_engine &operator=(matlab_engine &&e);

  matlab_engine(const matlab_engine &) = delete;
  matlab_engine &operator=(const matlab_engine &) = delete;

public:
  bool started() const;
  bool run(const std::string &cmd) const;
  std::string last_message() const;
  bool error_last_run() const;

  matlab_mxarray var(const std::string &name) const;
  bool set_var(const std::string &name, const matlab_mxarray &mxa);

  const matlab_engine &operator<<(const std::string &cmd) const;

  bool cd_and_add_all_subfolders(const std::string &dir);

private:
  char *_buffer;
  void *_eng;
};
}