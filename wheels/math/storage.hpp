#pragma once

#include <array>
#include <vector>

#include "../core/type.hpp"
#include "tensor_shape.hpp"

namespace wheels {
	
    template <class T, class NumelT> class basic_storage;

    template <class T, size_t Numel>
    class basic_storage<T, const_size<Numel>> {
    public:
        basic_storage() {}
        explicit basic_storage(const const_size<Numel> &) {}
        template <class IterT>
        basic_storage(IterT b, IterT e) {
            std::copy(b, e, _data.begin());
        }

        const T * data() const { return _data.data(); }
        T * data() { return _data.data(); }
        const T & operator[](size_t i) const { return _data[i]; }
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

        static constexpr auto is_continuous() { return yes() : }
        static constexpr auto is_parallel_writable() { return yes(); }
        static constexpr auto is_parallel_readable() { return yes(); }

    private:
        std::vector<decltype(uint_type_of_bytes(const_size<sizeof(bool)>()))::type> _data;
    };



}