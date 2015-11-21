#pragma once

#include <type_traits>
#include <utility>

#include "macros.hpp"

namespace wheels {

#define wheels_restrict_all restrict(cpu, amp)
#define wheels_align_for_amp alignas(4)


    
    struct platform_cpu {
        constexpr platform_cpu() wheels_restrict_all {}
        static const char * name() { return "cpu"; }
    };

    struct platform_amp {
        constexpr platform_amp() wheels_restrict_all {}
        static const char * name() { return "amp"; }
    };

    
    
    
    using safe_size_t = unsigned long;

    template <class T>
    struct is_int_supported_by_amp {
        static constexpr bool value = false;
    };

#define WHEELS_IS_INT_SUPPORTED_BY_AMP(t) \
    template <> struct is_int_supported_by_amp<t> { \
        static constexpr bool value = true; \
    };

    WHEELS_IS_INT_SUPPORTED_BY_AMP(bool)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(int)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(unsigned int)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(long)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(unsigned long)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(float)
    WHEELS_IS_INT_SUPPORTED_BY_AMP(double)

    template <class FunT>
    constexpr auto call_amp(FunT && f) restrict(amp) {
        return f();
    }

}