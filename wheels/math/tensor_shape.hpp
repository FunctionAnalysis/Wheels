#pragma once

#include "../core/constants.hpp"

namespace wheels {

    template <class ... SizeTs> class tensor_shape;

    using ignore_t = decltype(std::ignore);

    // tensor_shape
    template <>
    class tensor_shape<> {
    public:
        constexpr auto degree() const { return const_size<0>(); }
        constexpr auto magnitude() const { return const_size<1>(); }

        constexpr tensor_shape() {}

        constexpr size_t sub2ind() const { return 0u; }
        void ind2sub(size_t ind) const {}

        template <class SubsIterT> constexpr size_t sub2ind_by_iter(SubsIterT subs_iter) const { return 0u; }
        template <class SubsIterT> void ind2sub_by_iter(size_t ind, SubsIterT subs_iter) const {}

        template <class FunT, class ... Ts>
        void for_each_subscript(FunT && fun, Ts && ... args) const {
            fun(args ...);
        }
        template <class FunT, class ... Ts>
        constexpr bool for_each_subscript_if(FunT && fun, Ts && ... args) const {
            return fun(args ...);
        }
        template <size_t Idx, class FunT, class ... Ts>
        std::enable_if_t<Idx == 0> for_each_subscript_until(const const_index<Idx> &, FunT && fun, Ts && ... args) const {
            fun(args ...);
        }

        constexpr bool operator == (const tensor_shape &) const { return true; }

        template <class Archive> void serialize(Archive & ar) {}
    };


    template <size_t S, class ... SizeTs>
    class tensor_shape<const_size<S>, SizeTs ...> : public tensor_shape<SizeTs ...> {        
        using this_t = tensor_shape<const_size<S>, SizeTs ...>;
        using rest_tensor_shape_t = tensor_shape<SizeTs ...>;

    public:
        constexpr const rest_tensor_shape_t & rest() const { return (const rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t & rest() { return (rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t && rest_rref() { return (rest_tensor_shape_t &&)(*this); }

        constexpr auto degree() const { return const_size<sizeof...(SizeTs) + 1>(); }
        constexpr auto value() const { return const_size<S>(); }
        constexpr auto magnitude() const { return value() * rest().magnitude(); }

        // ctor
        constexpr tensor_shape() : rest_tensor_shape_t() {}

        // ctor from vals
        template <class ... Ts>
        constexpr explicit tensor_shape(const_size<S>, Ts ... vals) : rest_tensor_shape_t(vals ...) {}
        template <class T, class ... Ts>
        constexpr explicit tensor_shape(T v, Ts ... vals) : rest_tensor_shape_t(vals ...) {
            assert(v == S);
            static_assert(is_int<T>::value, "T must be an integer type");
        }
        template <class ... Ts>
        constexpr explicit tensor_shape(ignore_t, Ts ... vals) : rest_tensor_shape_t(vals ...) {}
      
        // ctor from tensor_shape
        template <class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        constexpr tensor_shape(const tensor_shape<const_size<S>, SizeT2s...> & t) : rest_tensor_shape_t(t.rest()) {}
        template <class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        constexpr tensor_shape(const tensor_shape<size_t, SizeT2s...> & t) : rest_tensor_shape_t(t.rest()) {
            assert(t.value() == S);
        }

        // =
        template <class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<const_size<S>, SizeT2s...> & t) {
            rest() = t.rest();
            return *this;
        }
        template <class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<size_t, SizeT2s...> & t) {
            assert(t.value() == S);
            rest() = t.rest();
            return *this;
        }


        // copy ctor
        tensor_shape(const tensor_shape &) = default;
        tensor_shape(tensor_shape &&) = default;
        tensor_shape & operator = (const tensor_shape &) = default;
        tensor_shape & operator = (tensor_shape &&) = default;


        // sub2ind
        constexpr size_t sub2ind() const { return 0u; }
        template <class T, class ... Ts>
        constexpr size_t sub2ind(T sub, Ts ... subs) const {
            return sub * rest().magnitude() + rest().sub2ind(subs...);
        }

        // ind2sub
        void ind2sub(size_t ind) const {}
        template <class T, class ... Ts>
        void ind2sub(size_t ind, T & sub, Ts & ... subs) const {
            const auto lm = rest().magnitude();
            sub = ind / lm;
            rest().ind2sub(ind % lm, subs ...);
        }

        // sub2ind_by_iter
        template <class SubsIterT>
        size_t sub2ind_by_iter(SubsIterT subs_iter) const {
            const size_t cur_sub = *subs_iter;
            ++subs_iter;
            return cur_sub * rest().magnitude() + rest().sub2ind_by_iter(subs_iter);
        }

        // ind2sub_by_iter
        template <class SubsIterT>
        void ind2sub_by_iter(size_t ind, SubsIterT subs_iter) const {
            const auto lm = rest().magnitude();
            *subs_iter = ind / lm;
            rest().ind2sub_by_iter(ind % lm, ++subs_iter);
        }

        template <size_t Idx> 
        constexpr auto at(const const_index<Idx> &) const { return rest().at(const_index<Idx - 1>()); }
        constexpr auto at(const const_index<0u> &) const { return value(); }
        template <class T, T Idx, bool _B = std::is_same<T, size_t>::value, wheels_enable_if(!_B)>
        constexpr auto at(const const_ints<T, Idx> &) const { return at(const_index<Idx>()); }

        template <size_t Idx> 
        void resize(const const_index<Idx> &, size_t ns) { rest().resize(const_index<Idx - 1>(), ns); }
        void resize(const const_index<0u> &, size_t ns) { assert(ns == S); }
        template <class T, T Idx, bool _B = std::is_same<T, size_t>::value, wheels_enable_if(!_B)>
        void resize(const const_ints<T, Idx> &, size_t ns) { resize(const_index<Idx>(), ns); }

        template <class FunT, class ... Ts>
        void for_each_subscript(FunT && fun, Ts && ... args) const {
            const auto n = value();
            for (size_t i = 0; i < n; i++) {
                rest().for_each_subscript(fun, args ..., i);
            }
        }

        template <class FunT, class ... Ts>
        bool for_each_subscript_if(FunT && fun, Ts && ... args) const {
            const auto n = value();
            size_t i = 0;
            for (; i < n; i++) {
                if (!rest().for_each_subscript_if(fun, args ..., i))
                    break;
            }
            return i == n;
        }

        template <size_t Idx, class FunT, class ... Ts>
        std::enable_if_t<Idx == sizeof...(SizeTs)+1> for_each_subscript_until(const const_index<Idx> &, FunT && fun, Ts && ... args) const {
            fun(args ...);
        }
        template <size_t Idx, class FunT, class ... Ts>
        std::enable_if_t<(Idx < sizeof...(SizeTs)+1)> for_each_subscript_until(const const_index<Idx> & idx, FunT && fun, Ts && ... args) const {
            const auto n = value();
            for (size_t i = 0; i < n; i++) {
                rest().for_each_subscript_until(idx, fun, args ..., i);
            }
        }

        template <class ... SizeT2s>
        constexpr bool operator == (const tensor_shape<const_size<S>, SizeT2s ...> & b) const {
            return rest() == b.rest();
        }
        template <class ... SizeT2s>
        bool operator == (const tensor_shape<size_t, SizeT2s...> & b) const {
            return value() == b.value() && rest() == b.rest();
        }
        
        template <class Archive> void serialize(Archive & ar) { ar(rest()); }
    };


    template <class ... SizeTs>
    class tensor_shape<size_t, SizeTs ...> : public tensor_shape<SizeTs ...> {
        using this_t = tensor_shape<size_t, SizeTs ...>;
        using rest_tensor_shape_t = tensor_shape<SizeTs ...>;

    public:
        constexpr const rest_tensor_shape_t & rest() const { return (const rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t & rest() { return (rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t && rest_rref() { return (rest_tensor_shape_t &&)(*this); }

        constexpr auto degree() const { return const_size<sizeof...(SizeTs)+1>(); }
        size_t value() const { return _val; }
        size_t magnitude() const { return _mag; }

        // ctor
        tensor_shape() : rest_tensor_shape_t(), _val(0), _mag(0) {}

        // ctor from vals
        template <class T, class ... Ts>
        explicit tensor_shape(T v, Ts ... vals) : rest_tensor_shape_t(vals ...), _val(v), _mag(v * rest().magnitude()) {
            static_assert(is_int<T>::value, "T must be an integer type");
        }

        // ctor from tensor_shape
        template <class ST, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape(const tensor_shape<ST, SizeT2s...> & t) : rest_tensor_shape_t(t.rest()), _val(t.value()), _mag(t.magnitude()) {}

        // =
        template <class ST, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<ST, SizeT2s...> & t) {
            _val = t.value(); _mag = t.magnitude();
            rest() = t.rest();
            return *this;
        }

        // copy ctor
        tensor_shape(const tensor_shape &) = default;
        tensor_shape(tensor_shape &&) = default;
        tensor_shape & operator = (const tensor_shape &) = default;
        tensor_shape & operator = (tensor_shape &&) = default;

        // sub2ind
        constexpr size_t sub2ind() const { return 0u; }
        template <class T, class ... Ts>
        constexpr size_t sub2ind(T sub, Ts ... subs) const {
            return sub * rest().magnitude() + rest().sub2ind(subs...);
        }

        // ind2sub
        void ind2sub(size_t ind) const {}
        template <class T, class ... Ts>
        void ind2sub(size_t ind, T & sub, Ts & ... subs) const {
            const auto lm = rest().magnitude();
            sub = ind / lm;
            rest().ind2sub(ind % lm, subs ...);
        }

        // sub2ind_by_iter
        template <class SubsIterT>
        size_t sub2ind_by_iter(SubsIterT subs_iter) const {
            const size_t cur_sub = *subs_iter;
            ++subs_iter;
            return cur_sub * rest().magnitude() + rest().sub2ind_by_iter(subs_iter);
        }

        // ind2sub_by_iter
        template <class SubsIterT>
        void ind2sub_by_iter(size_t ind, SubsIterT subs_iter) const {
            const auto lm = rest().magnitude();
            *subs_iter = ind / lm;
            rest().ind2sub_by_iter(ind % lm, ++subs_iter);
        }

        template <size_t Idx>
        constexpr auto at(const const_index<Idx> &) const { return rest().at(const_index<Idx - 1>()); }
        constexpr size_t at(const const_index<0u>) const { return _val; }
        template <class T, T Idx, bool _B = std::is_same<T, size_t>::value, wheels_enable_if(!_B)>
        constexpr auto at(const const_ints<T, Idx> &) const { return at(const_index<Idx>()); }

        template <size_t Idx> 
        void resize(const const_index<Idx> &, size_t ns) { 
            rest().resize(const_index<Idx - 1>(), ns); 
            _mag = _val * rest().magnitude();
        }
        void resize(const const_index<0u> &, size_t ns) { 
            _val = ns;
            _mag = _val * rest().magnitude();
        }
        template <class T, T Idx, bool _B = std::is_same<T, size_t>::value, wheels_enable_if(!_B)>
        void resize(const const_ints<T, Idx> &, size_t ns) { resize(const_index<Idx>(), ns); }


        template <class FunT, class ... Ts>
        void for_each_subscript(FunT && fun, Ts && ... args) const {
            const size_t n = value();
            for (size_t i = 0; i < n; i++) {
                rest().for_each_subscript(fun, args ..., i);
            }
        }

        template <class FunT, class ... Ts>
        bool for_each_subscript_if(FunT && fun, Ts && ... args) const {
            const size_t n = value();
            size_t i = 0;
            for (; i < n; i++) {
                if (!rest().for_each_subscript_if(fun, args ..., i))
                    break;
            }
            return i == n;
        }

        template <size_t Idx, class FunT, class ... Ts>
        std::enable_if_t<Idx == sizeof...(SizeTs)+1> for_each_subscript_until(const const_index<Idx> &, FunT && fun, Ts && ... args) const {
            fun(args ...);
        }
        template <size_t Idx, class FunT, class ... Ts>
        std::enable_if_t<(Idx < sizeof...(SizeTs)+1)> for_each_subscript_until(const const_index<Idx> & idx, FunT && fun, Ts && ... args) const {
            const size_t n = value();
            for (size_t i = 0; i < n; i++) {
                rest().for_each_subscript_until(idx, fun, args ..., i);
            }
        }

        template <class ST, class ... SizeT2s>
        bool operator == (const tensor_shape<ST, SizeT2s ...> & b) const {
            return value() == b.value() && rest() == b.rest();
        }

        template <class Archive> void serialize(Archive & ar) { ar(_val, _mag);  ar(rest()); }

    private:
        size_t _val;
        size_t _mag;
    };


    template <class ... SizeT1s, class ... SizeT2s>
    constexpr bool operator != (const tensor_shape<SizeT1s...> & s1, const tensor_shape<SizeT2s...> & s2) {
        return !(s1 == s2);
    }


    namespace details {
        template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
        constexpr size_t _to_size(const T & s) {
            return (size_t)s;
        }
        template <class T, T Val>
        constexpr auto _to_size(const const_ints<T, Val> &) {
            return const_size<Val>();
        }
    }

    // make_tensor_shape
    template <class ... SizeTs>
    constexpr auto make_tensor_shape(const SizeTs & ... sizes) {
        return tensor_shape<decltype(details::_to_size(sizes)) ...>(details::_to_size(sizes) ...);
    }

    // make_tensor_shape
    template <class T, T ... Sizes>
    constexpr auto make_tensor_shape(const const_ints<T, Sizes...> &) {
        return tensor_shape<const_ints<size_t, Sizes>...>();
    }


    // stream
    namespace details {
        template <class ShapeT, size_t ... Is>
        inline std::ostream & _stream_seq(std::ostream & os, const ShapeT & shape, std::index_sequence<Is...>) {
            return print(" ", os << "[", shape.at(const_index<Is>()) ...) << "]";
        }
    }
    template <class ... SizeTs>
    inline std::ostream & operator << (std::ostream & os, const tensor_shape<SizeTs...> & shape) {
        return details::_stream_seq(os, shape, std::make_index_sequence<sizeof...(SizeTs)>());
    }   


}