#pragma once

#include "constants.hpp"

namespace wheels {

	template <class T>
    struct type_obj {
        using type = T;
        
        constexpr type_obj() {}

        constexpr auto decay() const { return type_obj<std::decay_t<T>>(); }

        constexpr auto declval() const { return std::declval<T>(); }
        constexpr auto defaultval() const { return T(); }
        template <class ... ArgTs>
        constexpr auto construct(ArgTs && ... args) const {
            return T(std::forward<ArgTs>(args) ...);
        }

        template <class Archive> void serialize(Archive &) {}
    };

    template <class T>
    constexpr auto type_of(T && t) {
        return type_obj<T &&>();
    }

    template <class T1, class T2>
    constexpr auto operator == (const type_obj<T1> &, const type_obj<T2> &) {
        return const_bool<std::is_same<T1, T2>::value>();
    }

    template <class T1, class T2>
    constexpr auto operator != (const type_obj<T1> &, const type_obj<T2> &) {
        return const_bool<!std::is_same<T1, T2>::value>();
    }





    namespace details {
        template <size_t Bytes> struct _int_of {};
        template <size_t Bytes> struct _uint_of {};

        template <> struct _int_of<1> { using type = int8_t; };
        template <> struct _uint_of<1> { using type = uint8_t; };
        template <> struct _int_of<2> { using type = int16_t; };
        template <> struct _uint_of<2> { using type = uint16_t; };
        template <> struct _int_of<4> { using type = int32_t; };
        template <> struct _uint_of<4> { using type = uint32_t; };
        template <> struct _int_of<8> { using type = int64_t; };
        template <> struct _uint_of<8> { using type = uint64_t; };
    }

    // int_type_of_bytes
    template <class T, T Bytes>
    constexpr auto int_type_of_bytes(const const_ints<T, Bytes> &) {
        return type_obj<typename details::_int_of<Bytes>::type>();
    }

    // uint_type_of_bytes
    template <class T, T Bytes>
    constexpr auto uint_type_of_bytes(const const_ints<T, Bytes> &) {
        return type_obj<typename details::_uint_of<Bytes>::type>();
    }


}