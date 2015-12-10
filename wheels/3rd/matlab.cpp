#include <engine.h>

#include "matlab.hpp"
#include <string>

namespace wheels {

matlab_mxarray::matlab_mxarray()
    : _m(nullptr), _destroy_when_out_of_scope(false) {}
matlab_mxarray::matlab_mxarray(mxArray *m, bool dos)
    : _m(m), _destroy_when_out_of_scope(dos) {}
matlab_mxarray::matlab_mxarray(const matlab_mxarray &mx) {
  if (mx._m) {
    _m = mxDuplicateArray(mx._m);
  } else {
    _m = nullptr;
  }
  _destroy_when_out_of_scope = mx._destroy_when_out_of_scope;
}
matlab_mxarray::matlab_mxarray(matlab_mxarray &&mx)
    : _m(mx._m), _destroy_when_out_of_scope(mx._destroy_when_out_of_scope) {
  mx._m = nullptr;
  mx._destroy_when_out_of_scope = false;
}

matlab_mxarray::~matlab_mxarray() {
  if (_destroy_when_out_of_scope) {
    mxDestroyArray(_m);
    _m = nullptr;
  }
}

matlab_mxarray &matlab_mxarray::operator=(const matlab_mxarray &mx) {
  if (mx._m) {
    _m = mxDuplicateArray(mx._m);
  } else {
    _m = nullptr;
  }
  _destroy_when_out_of_scope = mx._destroy_when_out_of_scope;
  return *this;
}

matlab_mxarray &matlab_mxarray::operator=(matlab_mxarray &&mx) {
  std::swap(_m, mx._m);
  std::swap(_destroy_when_out_of_scope, mx._destroy_when_out_of_scope);
  return *this;
}

mxArray *matlab_mxarray::mxa() const { return _m; }

bool matlab_mxarray::is_numeric() const { return mxIsNumeric(_m); }
bool matlab_mxarray::is_cell() const { return mxIsCell(_m); }
bool matlab_mxarray::is_logical() const { return mxIsLogical(_m); }
bool matlab_mxarray::is_char() const { return mxIsChar(_m); }
bool matlab_mxarray::is_struct() const { return mxIsStruct(_m); }
bool matlab_mxarray::is_opaque() const { return mxIsOpaque(_m); }
bool matlab_mxarray::is_function_handle() const {
  return mxIsFunctionHandle(_m);
}
bool matlab_mxarray::is_object() const { return mxIsObject(_m); }
bool matlab_mxarray::is_complex() const { return mxIsComplex(_m); }
bool matlab_mxarray::is_sparse() const { return mxIsSparse(_m); }
bool matlab_mxarray::is_double() const { return mxIsDouble(_m); }
bool matlab_mxarray::is_single() const { return mxIsSingle(_m); }
bool matlab_mxarray::is_int8() const { return mxIsInt8(_m); }
bool matlab_mxarray::is_uint8() const { return mxIsUint8(_m); }
bool matlab_mxarray::is_int16() const { return mxIsInt16(_m); }
bool matlab_mxarray::is_uint16() const { return mxIsUint16(_m); }
bool matlab_mxarray::is_int32() const { return mxIsInt32(_m); }
bool matlab_mxarray::is_uint32() const { return mxIsUint32(_m); }
bool matlab_mxarray::is_int64() const { return mxIsInt64(_m); }
bool matlab_mxarray::is_uint64() const { return mxIsUint64(_m); }

size_t matlab_mxarray::m() const { return mxGetM(_m); }

size_t matlab_mxarray::n() const { return mxGetN(_m); }

void matlab_mxarray::set_m(size_t m) { mxSetM(_m, m); }

void matlab_mxarray::set_n(size_t n) { mxSetN(_m, n); }

bool matlab_mxarray::empty() const { return mxIsEmpty(_m); }

bool matlab_mxarray::is_from_global_workspace() const {
  return mxIsFromGlobalWS(_m);
}

void matlab_mxarray::set_is_from_global_workspace(bool b) {
  mxSetFromGlobalWS(_m, b);
}

size_t matlab_mxarray::numel() const { return mxGetNumberOfElements(_m); }

std::string matlab_mxarray::to_string() const {
  size_t len = numel() + 1;
  char *buffer = new char[len];
  memset(buffer, '\0', len);
  mxGetString(_m, buffer, len);
  std::string str = buffer;
  delete[] buffer;
  return str;
}

matlab_mxarray matlab_mxarray::cell(size_t i) const { return mxGetCell(_m, i); }

void matlab_mxarray::set_cell(size_t i, const matlab_mxarray &a) {
  mxSetCell(_m, i, a._m);
}

int matlab_mxarray::nfields() const { return mxGetNumberOfFields(_m); }

const char *matlab_mxarray::field_name(int n) const {
  return mxGetFieldNameByNumber(_m, n);
}

int matlab_mxarray::field_number(const std::string &name) const {
  return mxGetFieldNumber(_m, name.c_str());
}

matlab_mxarray matlab_mxarray::field(const std::string &name, int i) const {
  return mxGetField(_m, i, name.c_str());
}

void matlab_mxarray::set_field(const std::string &name, const matlab_mxarray &a,
                               int i) const {
  mxSetField(_m, i, name.c_str(), a._m);
}

matlab_mxarray matlab_mxarray::property(const std::string &name,
                                        size_t i) const {
  return mxGetProperty(_m, i, name.c_str());
}

void matlab_mxarray::set_property(const std::string &name, int i,
                                  const matlab_mxarray &a) {
  mxSetProperty(_m, i, name.c_str(), a._m);
}

const char *matlab_mxarray::class_name() const { return mxGetClassName(_m); }

size_t matlab_mxarray::ndims() const { return mxGetNumberOfDimensions(_m); }

std::vector<size_t> matlab_mxarray::dims() const {
  std::vector<size_t> ds(ndims());
  std::copy_n(mxGetDimensions(_m), ds.size(), ds.begin());
  return ds;
}

size_t matlab_mxarray::dim(size_t d) const { return mxGetDimensions(_m)[d]; }

size_t matlab_mxarray::length() const {
  auto ds = dims();
  if (ds.empty())
    return 0;
  return *std::max_element(ds.begin(), ds.end());
}

size_t matlab_mxarray::single_sub(size_t dim, size_t *subs) const {
  return mxCalcSingleSubscript(_m, dim, subs);
}

void *matlab_mxarray::real_data() const { return mxGetData(_m); }

void *matlab_mxarray::imag_data() const { return mxGetImagData(_m); }

matlab_matfile::matlab_matfile() : _fp(nullptr) {}
matlab_matfile::matlab_matfile(const std::string &fname,
                               const std::string &mode)
    : _fname(fname) {
  _fp = matOpen(fname.c_str(), mode.c_str());
}
matlab_matfile::matlab_matfile(const std::string &fname, OpeningMode mode)
    : _fname(fname), _fp(nullptr) {
  std::string m;
  switch (mode) {
  case Read:
    m = "r";
    break;
  case Update:
    m = "u";
    break;
  case Write:
    m = "w";
    break;
  case Write_4:
    m = "w4";
    break;
  case Write_CompressedData:
    m = "wz";
    break;
  case Write_7_3:
    m = "w7.3";
    break;
  default:
    return;
  }
  _fp = matOpen(fname.c_str(), m.c_str());
}

matlab_matfile::matlab_matfile(matlab_matfile &&a) {
  _fname = a._fname;
  a._fname.clear();
  _fp = a._fp;
  a._fp = nullptr;
}
matlab_matfile &matlab_matfile::operator=(matlab_matfile &&a) {
  std::swap(_fname, a._fname);
  std::swap(_fp, a._fp);
  return *this;
}

matlab_matfile::~matlab_matfile() {
  if (_fp) {
    matClose(static_cast<::MATFile *>(_fp));
    _fp = nullptr;
  }
}

std::vector<std::string> matlab_matfile::var_names() const {
  int num = 0;
  char **names = matGetDir(static_cast<::MATFile *>(_fp), &num);
  if (!names)
    return std::vector<std::string>();
  std::vector<std::string> vnames(num);
  for (int i = 0; i < num; i++) {
    vnames[i] = names[i];
  }
  mxFree(names);
  return vnames;
}

matlab_mxarray matlab_matfile::var(const std::string &name) const {
  return matGetVariable(static_cast<::MATFile *>(_fp), name.c_str());
}

bool matlab_matfile::set_var(const std::string &name, const matlab_mxarray &mxa,
                             bool asGlobal /*= false*/) {
  if (!asGlobal) {
    return matPutVariable(static_cast<::MATFile *>(_fp), name.c_str(),
                          static_cast<mxArray *>(mxa.mxa())) == 0;
  } else {
    return matPutVariableAsGlobal(static_cast<::MATFile *>(_fp), name.c_str(),
                                  static_cast<mxArray *>(mxa.mxa())) == 0;
  }
}

bool matlab_matfile::remove_var(const std::string &name) {
  return matDeleteVariable(static_cast<::MATFile *>(_fp), name.c_str()) == 0;
}

matlab_engine::matlab_engine(const std::string &defaultDir, bool singleUse)
    : _eng(nullptr), _buffer(nullptr) {
  static const int bufferSize = 1024;
  if (singleUse) {
    _eng = engOpenSingleUse(nullptr, nullptr, nullptr);
  } else {
    _eng = engOpen(nullptr);
  }
  if (_eng) {
    engSetVisible(static_cast<::Engine *>(_eng), false);
    _buffer = new char[bufferSize];
    std::memset(_buffer, 0, bufferSize);
    engOutputBuffer(static_cast<::Engine *>(_eng), _buffer, bufferSize);
    std::cout << "matlab_engine Engine Launched" << std::endl;
    if (!defaultDir.empty()) {
      engEvalString(static_cast<::Engine *>(_eng),
                    ("cd " + defaultDir + "; startup; pwd").c_str());
    }
    std::cout << _buffer << std::endl;
  }
}

matlab_engine::~matlab_engine() {
  if (_eng) {
    engClose(static_cast<::Engine *>(_eng));
    std::cout << "matlab_engine Engine Closed" << std::endl;
  }
  delete[] _buffer;
  _eng = nullptr;
  _buffer = nullptr;
}

matlab_engine::matlab_engine(matlab_engine &&e) {
  _eng = e._eng;
  e._eng = nullptr;
  _buffer = e._buffer;
  e._buffer = nullptr;
}

matlab_engine &matlab_engine::operator=(matlab_engine &&e) {
  std::swap(e._eng, _eng);
  std::swap(e._buffer, _buffer);
  return *this;
}

bool matlab_engine::started() const { return _eng != nullptr; }

bool matlab_engine::run(const std::string &cmd) const {
  bool ret = engEvalString(static_cast<::Engine *>(_eng), cmd.c_str()) == 0;
  if (strlen(_buffer) > 0) {
    std::cout << "[Message when executing '" << cmd << "']:\n" << _buffer
              << std::endl;
  }
  return ret;
}

std::string matlab_engine::last_message() const { return _buffer; }

bool matlab_engine::error_last_run() const {
  return std::string(_buffer).substr(0, 5) == "Error";
}

matlab_mxarray matlab_engine::var(const std::string &name) const {
  return matlab_mxarray(
      engGetVariable(static_cast<::Engine *>(_eng), name.c_str()), true);
}

bool matlab_engine::set_var(const std::string &name,
                            const matlab_mxarray &mxa) {
  return engPutVariable(static_cast<::Engine *>(_eng), name.c_str(),
                        static_cast<mxArray *>(mxa.mxa())) == 0;
}

const matlab_engine &matlab_engine::operator<<(const std::string &cmd) const {
  run(cmd);
  return *this;
}

bool matlab_engine::cd_and_add_all_subfolders(const std::string &dir) {
  return run("cd " + dir) && run("addpath(genpath('.'));");
}
}