#pragma once

#include "constants.hpp"

namespace wheels {

	template <class T>
    struct type_obj {
        using type = T;
        
        constexpr auto decay() const { return type_obj<std::decay_t<T>>(); }

        constexpr auto declval() const { return std::declval<T>(); }
        constexpr auto defaultval() const { return T(); }
        constexpr auto zeroval() const { return T(); }

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

}