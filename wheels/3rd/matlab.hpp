#pragma once

#include <complex>

#include <mex.h>
#include <mat.h>

#include "../tensor/functions.hpp"
#include "../tensor/matrices.hpp"

namespace wheels {

    namespace details {

        template <class T>
        struct mx_type_to_class_id {
            static const mxClassID value = mxUNKNOWN_CLASS;
        };
        template <mxClassID ClassID>
        struct mx_class_id_to_type {
            using type = void;
        };

#define WHEELS_BIND_TYPE_WITH_MX_CLASSID(T, id) \
    template <> struct mx_type_to_class_id<T> { static const mxClassID value = id; }; \
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

        template <class T>
        struct is_complex : no {};
        template <class T>
        struct is_complex<std::complex<T>> : yes {};

        template <class T>
        struct real_component {
            using type = void;
        };
        template <class T>
        struct real_component<std::complex<T>> {
            using type = T;
        };

        template <class ShapeT, class DPT, size_t ... Is>
        mxArray * _make_mx_array_seq(const tensor_category<ShapeT, DPT> & ts, const no & cplx, 
            const const_ints<size_t, Is...> &) {
            using value_t = typename DPT::value_type;
            static_assert(!is_complex<value_t>::value, "invalid type");
            static const mxClassId classID = mx_type_to_class_id<value_t>::value;
            static_assert(classID != mxUNKNOWN_CLASS, "type not supported in MATLAB");

            static const size_t rank = ShapeT::rank;
            const size_t dims[] = { ts.shape().at(const_index<Is>()) ... };
            mxArray * mx = mxCreateNumericArray(rank, dims, classID, mxREAL);

            value_t * data = static_cast<value_t *>(mxGetData(mx));
            for_each_subscript(ts.shape(), [&data, &ts, &mx](auto && ... subs) {
                size_t subs_data[] = { static_cast<size_t>(subs) ... };
                size_t offset = mxCalcSingleSubscript(mx, rank, subs_data);
                data[offset] = ts.at_subs_const(subs ...);
            });
            return mx;
        }

        template <class ShapeT, class DPT, size_t ... Is>
        mxArray * _make_mx_array_seq(const tensor_category<ShapeT, DPT> & ts, const yes & cplx, 
            const const_ints<size_t, Is...> &) {
            using value_t = typename DPT::value_type;
            static_assert(is_complex<value_t>::value, "invalid type");
            using real_value_t = typename real_component<value_t>::type;
            static const mxClassId classID = mx_type_to_class_id<real_value_t>::value;
            static_assert(classID != mxUNKNOWN_CLASS, "type not supported in MATLAB");

            static const size_t rank = ShapeT::rank;
            const size_t dims[] = { ts.shape().at(const_index<Is>()) ... };
            mxArray * mx = mxCreateNumericArray(rank, dims, classID, mxCOMPLEX);

            value_t * real_data = static_cast<value_t *>(mxGetData(mx));
            value_t * imag_data = static_cast<value_t *>(mxGetImagData(mx));
            for_each_subscript(ts.shape(), [&real_data, &imag_data, &ts, &mx](auto && ... subs) {
                size_t subs_data[] = { static_cast<size_t>(subs) ... };
                size_t offset = mxCalcSingleSubscript(mx, rank, subs_data);
                auto e = ts.at_subs_const(subs ...);
                real_data[offset] = e.real();
                imag_data[offset] = e.imag();
            });
            return mx;
        }

    }


    template <class ShapeT, class DPT>
    inline mxArray * make_mx_array(const tensor_category<ShapeT, DPT> & ts) {
        return details::_make_mx_array_seq(ts, details::is_complex<typename DPT::value_type>(), 
            make_rank_sequence(ts.shape()));
    }



    class matlab_mxarray {
    public:
        matlab_mxarray();
        explicit matlab_mxarray(mxArray * m, bool dos = false);
        matlab_mxarray(const matlab_mxarray & mx);
        matlab_mxarray(matlab_mxarray && mx);
        ~matlab_mxarray();
        
        matlab_mxarray & operator = (const matlab_mxarray & mx);
        matlab_mxarray & operator = (matlab_mxarray && mx);

        template <class ShapeT, class DPT>
        matlab_mxarray(const tensor_category<ShapeT, DPT> & ts) 
            : _m(make_mx_array(ts)), _destroy_when_out_of_scope(true) {}

    public:
        mxArray * data() const;



    private:
        mxArray * _m;
        bool _destroy_when_out_of_scope;
    };

}