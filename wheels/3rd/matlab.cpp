#include "matlab.hpp"

namespace wheels {

    matlab_mxarray::matlab_mxarray() : _m(nullptr), _destroy_when_out_of_scope(false) {}
    matlab_mxarray::matlab_mxarray(mxArray * m, bool dos) : _m(m), _destroy_when_out_of_scope(dos) {}
    matlab_mxarray::matlab_mxarray(const matlab_mxarray & mx) {
        if (mx._m) {
            _m = mxDuplicateArray(mx._m);
        } else {
            _m = nullptr;
        }
        _destroy_when_out_of_scope = mx._destroy_when_out_of_scope;
    }
    matlab_mxarray::matlab_mxarray(matlab_mxarray && mx) : _m(mx._m), _destroy_when_out_of_scope(mx._destroy_when_out_of_scope){
        mx._m = nullptr;
        mx._destroy_when_out_of_scope = false;
    }

    matlab_mxarray::~matlab_mxarray() {
        if (_destroy_when_out_of_scope) {
            mxDestroyArray(_m);
            _m = nullptr;
        }
    }

    matlab_mxarray & matlab_mxarray::operator=(const matlab_mxarray & mx) {
        if (mx._m) {
            _m = mxDuplicateArray(mx._m);
        } else {
            _m = nullptr;
        }
        _destroy_when_out_of_scope = mx._destroy_when_out_of_scope;
        return *this;
    }

    matlab_mxarray & matlab_mxarray::operator=(matlab_mxarray && mx) {
        std::swap(_m, mx._m);
        std::swap(_destroy_when_out_of_scope, mx._destroy_when_out_of_scope);
        return *this;
    }

    mxArray * matlab_mxarray::data() const {
        return _m;
    }

}