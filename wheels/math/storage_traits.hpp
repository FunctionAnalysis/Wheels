#pragma once

#include <array>
#include <vector>

#include "../core/types.hpp"

#include "tensor_shape.hpp"
#include "functors.hpp"

namespace wheels {


    


    template <class T>
    struct storage_traits {
        using value_type = typename T::value_type;
        using platform = platform_cpu;

        static constexpr auto is_default_constructible() { 
            return const_bool<std::is_default_constructible<T>::value>(); 
        }
        static constexpr T construct() {
            static_assert(false, "");
        }
        template <class ... ArgTs>
        static constexpr T construct_with_size(size_t sz, ArgTs && ... args) {
            static_assert(always<bool, false, ArgTs ...>::value, "");
        }

        static constexpr const value_type & element(const T & st, size_t i) {
            return st[i];
        }
        static value_type & element(T & st, size_t i) {
            return st[i];
        }
    };


    template <class T, size_t N>
    struct storage_traits<std::array<T, N>> {
        using value_type = T;
        using platform = platform_cpu;

        static constexpr yes is_default_constructible() {
            return yes();
        }

        static constexpr std::array<T, N> construct() {
            return std::array<T, N>();
        }
        static constexpr std::array<T, N> construct_with_size(size_t sz) {
            return std::array<T, N>();
        }
        template <class ... EleTs>
        static constexpr std::array<T, N> construct_with_elements(EleTs && ... eles) {
            return{ { (T)std::forward<EleTs>(eles) ... } };
        }
        template <class ... EleTs>
        static constexpr std::array<T, N> construct_with_size_elements(size_t sz, EleTs && ... eles) {
            return{ {(T)std::forward<EleTs>(eles) ...} };
        }


        static constexpr const value_type & element(const std::array<T, N> & st, size_t i) {
            return st[i];
        }
        static value_type & element(std::array<T, N> & st, size_t i) {
            return st[i];
        }
    };


    template <class T, class AllocT>
    struct storage_traits<std::vector<T, AllocT>> {
        using value_type = T;
        using platform = platform_cpu;

        static constexpr yes is_default_constructible() {
            return yes();
        }

        template <class ... ArgTs>
        static std::vector<T, AllocT> construct(ArgTs && ... args) {
            return std::vector<T, AllocT>(std::forward<ArgTs>(args) ...);
        }

        static std::vector<T, AllocT> construct_with_size(size_t sz) {
            return std::vector<T, AllocT>(sz);
        }
        template <class ... ArgTs>
        static std::vector<T, AllocT> construct_with_size(size_t sz, ArgTs && ... eles) {
            return { (T)std::forward<ArgTs>(eles) ... };
        }

        template <class ... EleTs>
        static std::vector<T, AllocT> construct_from_elements(EleTs && ... es) {
            return std::vector<T, AllocT>({(T)es ...});
        }

        static const value_type & element(const std::vector<T, AllocT> & st, size_t i) {
            return st[i];
        }
        static value_type & element(std::vector<T, AllocT> & st, size_t i) {
            return st[i];
        }
    };




    //template <class T>
    //struct storage_traits<const T> {
    //    using value_type = typename T::value_type;
    //    using platform = platform_cpu;

    //    static constexpr auto is_default_constructible() {
    //        return const_bool<std::is_default_constructible<T>::value>();
    //    }
    //    static constexpr auto construct_from_size(size_t sz) {
    //        return const T(sz);
    //    }
    //    template <class ... EleTs>
    //    static constexpr auto construct_from_elements(EleTs && ... es) {
    //        return const T({ es ... });
    //    }

    //    static constexpr const value_type & element(const T & st, size_t i) {
    //        return st[i];
    //    }
    //};



    /*



	
    template <class T, class NumelT> class basic_storage;

    template <class T, size_t Numel>
    class basic_storage<T, const_size<Numel>> {
    public:
        constexpr basic_storage() {}
        constexpr explicit basic_storage(const const_size<Numel> &) {}
        template <class IterT>
        basic_storage(IterT b, IterT e) {
            std::copy(b, e, _data.begin());
        }

        constexpr const T * data() const { return _data.data(); }
        T * data() { return _data.data(); }
        constexpr const T & operator[](size_t i) const { return _data[i]; }
        T & operator[](size_t i) { return _data[i]; }
        constexpr auto size() const { return const_size<Numel>(); }

        void resize(size_t sz) {}

        static constexpr auto is_continuous() {return yes(): }
        static constexpr auto is_parallel_writable() { return yes(); }
        static constexpr auto is_parallel_readable() { return yes(); }

    private:
        std::array<T, Numel> _data;
    };

    template <class T>
    class basic_storage<T, size_t> {
    public:
        basic_storage() {}
        explicit basic_storage(size_t sz) : _data(sz) {}
        template <class IterT> basic_storage(IterT b, IterT e) : _data(b, e) {}

        const T * data() const { return _data.data(); }
        T * data() { return _data.data(); }
        const T & operator[](size_t i) const { return _data[i]; }
        T & operator[](size_t i) { return _data[i]; }
        size_t size() const { return _data.size(); }

        void resize(size_t sz) { _data.resize(sz); }

        static constexpr auto is_continuous() { return yes() : }
        static constexpr auto is_parallel_writable() { return yes(); }
        static constexpr auto is_parallel_readable() { return yes(); }

    private:
        std::vector<T> _data;
    };

    template <>
    class basic_storage<bool, size_t> {
    public:
        basic_storage() {}
        explicit basic_storage(size_t sz) : _data(sz) {}
        template <class IterT> basic_storage(IterT b, IterT e) : _data(b, e) {}

        const bool * data() const { return (const bool *)(_data.data()); }
        bool * data() { return (bool*)(_data.data()); }
        bool operator[](size_t i) const { return _data[i]; }
        bool & operator[](size_t i) { return (bool&)_data[i]; }
        size_t size() const { return _data.size(); }

        void resize(size_t sz) { _data.resize(sz); }

        static constexpr auto is_continuous() { return yes(); }
        static constexpr auto is_parallel_writable() { return yes(); }
        static constexpr auto is_parallel_readable() { return yes(); }

    private:
        std::vector<type_t(uint_type_of_bytes(const_size<sizeof(bool)>()))> _data;
    };

*/

}