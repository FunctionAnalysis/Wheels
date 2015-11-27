#pragma once

#include "../core/constants.hpp"

namespace wheels {

    template <class T, class ... SizeTs> class tensor_shape;

    using ignore_t = decltype(std::ignore);

    // tensor_shape
    template <class T>
    class tensor_shape<T> {
        static_assert(std::is_integral<T>::value, "T should be an integral type");
    public:
        using value_type = T;
        static constexpr size_t rank = 0;
        static constexpr bool is_static = true;
        static constexpr size_t static_size_num = 0;

        constexpr const_ints<T, 1> magnitude() const { return const_ints<T, 1>(); }

        constexpr tensor_shape() {}
        template <class K> 
        constexpr tensor_shape(const tensor_shape<K> &) {}

        template <class Archive> void serialize(Archive & ar) {}
    };


    template <class T, T S, class ... SizeTs>
    class tensor_shape<T, const_ints<T, S>, SizeTs ...> : public tensor_shape<T, SizeTs ...> {   
        static_assert(std::is_integral<T>::value, "T should be an integral type");
        using this_t = tensor_shape<T, const_ints<T, S>, SizeTs ...>;
        using rest_tensor_shape_t = tensor_shape<T, SizeTs ...>;

    public:
        using value_type = T;
        static constexpr size_t rank = sizeof...(SizeTs)+1;
        static constexpr bool is_static = rest_tensor_shape_t::is_static;
        static constexpr size_t static_size_num = rest_tensor_shape_t::static_size_num + 1;

        const rest_tensor_shape_t & rest() const { return (const rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t & rest() { return (rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t && rest_rref() { return (rest_tensor_shape_t &&)(*this); }    

        constexpr const_ints<T, S> value() const { return const_ints<T, S>(); }
        constexpr auto magnitude() const { return value() * rest().magnitude(); }

        // ctor
        constexpr tensor_shape() : rest_tensor_shape_t() {}

        // ctor from vals
        template <class ... Ks>
        constexpr explicit tensor_shape(const const_ints<T, S> &, const Ks & ... vals) : rest_tensor_shape_t(vals ...) {}
        template <class K, class ... Ks>
        constexpr explicit tensor_shape(const K & v, const Ks & ... vals) : rest_tensor_shape_t(vals ...) {
            static_assert(is_int<K>::value, "T must be an integer type");
        }
        template <class ... Ks>
        constexpr explicit tensor_shape(ignore_t, const Ks & ... vals) : rest_tensor_shape_t(vals ...) {}
      
        // ctor from tensor_shape
        template <class K, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        constexpr tensor_shape(const tensor_shape<K, const_ints<K, S>, SizeT2s...> & t) : rest_tensor_shape_t(t.rest()) {}
        template <class K, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        constexpr tensor_shape(const tensor_shape<K, K, SizeT2s...> & t) : rest_tensor_shape_t(t.rest()) {}

        // =
        template <class K, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<K, const_ints<K, S>, SizeT2s...> & t) {
            rest() = t.rest();
            return *this;
        }
        template <class K, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<K, K, SizeT2s...> & t) {
            assert(t.value() == S);
            rest() = t.rest();
            return *this;
        }


        // copy ctor
        constexpr tensor_shape(const tensor_shape &) = default;
        tensor_shape(tensor_shape &&) = default;
        tensor_shape & operator = (const tensor_shape &) = default;
        tensor_shape & operator = (tensor_shape &&) = default;

       

        // at
        template <size_t Idx> 
        constexpr auto at(const const_index<Idx> &) const { 
            static_assert(Idx < rank, "Idx too large");
            return rest().at(const_index<Idx - 1>()); 
        }
        constexpr auto at(const const_index<0> &) const { return value(); }
        template <class K, K Idx, bool _B = std::is_same<K, size_t>::value, wheels_enable_if(!_B)>
        constexpr auto at(const const_ints<K, Idx> &) const { 
            return at(const_index<Idx>()); 
        }
        template <class K, K Idx>
        constexpr auto operator[](const const_ints<K, Idx> & i) const { return at(i); }


        // resize
        template <size_t Idx> 
        void resize(const const_index<Idx> &, T ns) { 
            rest().resize(const_index<Idx - 1>(), ns);
        }
        void resize(const const_index<0u> &, T ns) { 
            assert(ns == S); 
        }
        template <class K, K Idx, bool _B = std::is_same<K, size_t>::value, wheels_enable_if(!_B)>
        void resize(const const_ints<K, Idx> &, T ns) { 
            resize(const_index<Idx>(), ns);
        }
        
        template <class Archive> void serialize(Archive & ar) { ar(rest()); }
    };


    template <class T, class ... SizeTs>
    class tensor_shape<T, T, SizeTs ...> : public tensor_shape<T, SizeTs ...> {
        static_assert(std::is_integral<T>::value, "T should be an integral type");
        using this_t = tensor_shape<T, T, SizeTs ...>;
        using rest_tensor_shape_t = tensor_shape<T, SizeTs ...>;

    public:
        using value_type = T;
        static constexpr size_t rank = sizeof...(SizeTs)+1;
        static constexpr bool is_static = false;
        static constexpr size_t static_size_num = rest_tensor_shape_t::static_size_num;

        constexpr const rest_tensor_shape_t & rest() const { return (const rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t & rest() { return (rest_tensor_shape_t &)(*this); }
        rest_tensor_shape_t && rest_rref() { return (rest_tensor_shape_t &&)(*this); }

        constexpr T value() const { return _val; }
        constexpr T magnitude() const { return _mag; }

        // ctor
        constexpr tensor_shape() : rest_tensor_shape_t(), _val(0), _mag(0) {}

        // ctor from vals
        template <class K, class ... Ks>
        constexpr explicit tensor_shape(const K & v, const Ks & ... vals) 
            : rest_tensor_shape_t(vals ...), _val(v), _mag(v * rest().magnitude()) {
            static_assert(is_int<K>::value, "T must be an integer type");
        }

        // ctor from tensor_shape
        template <class K, class ST, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        constexpr tensor_shape(const tensor_shape<K, ST, SizeT2s...> & t) 
            : rest_tensor_shape_t(t.rest()), _val(t.value()), _mag(t.magnitude()) {
        }

        // =
        template <class K, class ST, class ... SizeT2s, class = std::enable_if_t<sizeof...(SizeT2s) == sizeof...(SizeTs)>>
        tensor_shape & operator = (const tensor_shape<K, ST, SizeT2s...> & t) {
            _val = t.value(); _mag = t.magnitude();
            rest() = t.rest();
            return *this;
        }

        // copy ctor
        constexpr tensor_shape(const tensor_shape &) = default;
        tensor_shape(tensor_shape &&) = default;
        tensor_shape & operator = (const tensor_shape &) = default;
        tensor_shape & operator = (tensor_shape &&) = default;


        // at
        template <size_t Idx>
        constexpr auto at(const const_index<Idx> &) const { 
            static_assert(Idx < rank, "Idx too large");
            return rest().at(const_index<Idx - 1>());
        }
        constexpr T at(const const_index<0u>) const { return _val; }
        template <class K, K Idx, bool _B = std::is_same<K, size_t>::value, wheels_enable_if(!_B)>
        constexpr auto at(const const_ints<K, Idx> &) const { 
            return at(const_index<Idx>());
        }
        template <class K, K Idx>
        constexpr auto operator[](const const_ints<K, Idx> & i) const { return at(i); }


        // resize
        template <size_t Idx> 
        void resize(const const_index<Idx> &, T ns) { 
            rest().resize(const_index<Idx - 1>(), ns); 
            _mag = _val * rest().magnitude();
        }
        void resize(const const_index<0u> &, T ns) { 
            _val = ns;
            _mag = _val * rest().magnitude();
        }
        template <class K, K Idx, bool _B = std::is_same<K, size_t>::value, wheels_enable_if(!_B)>
        void resize(const const_ints<K, Idx> &, size_t ns) { 
            resize(const_index<Idx>(), ns); 
        }

        template <class Archive> void serialize(Archive & ar) { ar(_val, _mag);  ar(rest()); }

    private:
        T _val;
        T _mag;
    };






    // sub2ind
    template <class T>
    constexpr T sub2ind(const tensor_shape<T> &) { 
        return 0; 
    }
    template <class T, class SizeT, class ... SizeTs>
    constexpr T sub2ind(const tensor_shape<T, SizeT, SizeTs...> &) { 
        return 0; 
    }
    template <class T, class SizeT, class ... SizeTs, class K, class ... Ks>
    constexpr T sub2ind(const tensor_shape<T, SizeT, SizeTs...> & shape, K sub, Ks ... subs) {
        return sub * shape.rest().magnitude() + sub2ind(shape.rest(), subs...);
    }


    // ind2sub
    template <class T, class IndexT>
    void ind2sub(const tensor_shape<T> &, const IndexT & ind) {}
    template <class T, class IndexT, class SizeT, class ... SizeTs>
    void ind2sub(const tensor_shape<T, SizeT, SizeTs...> &, const IndexT & ind) {}
    template <class T, class IndexT, class SizeT, class ... SizeTs, class K, class ... Ks>
    void ind2sub(const tensor_shape<T, SizeT, SizeTs...> & shape, const IndexT & ind, K & sub, Ks & ... subs) {
        const auto lm = shape.rest().magnitude();
        sub = ind / lm;
        ind2sub(shape.rest(), ind % lm, subs ...);
    }

    

    // sub2ind_by_iter
    template <class T, class SubsIterT> 
    constexpr T sub2ind_by_iter(const tensor_shape<T> &, SubsIterT subs_iter) { 
        return 0; 
    }
    template <class SubsIterT, class T, class SizeT, class ... SizeTs>
    T sub2ind_by_iter(const tensor_shape<T, SizeT, SizeTs...> & shape, SubsIterT subs_iter) {
        const auto cur_sub = *subs_iter;
        ++subs_iter;
        return cur_sub * shape.rest().magnitude() + sub2ind_by_iter(shape.rest(), subs_iter);
    }


    // ind2sub_by_iter
    template <class T, class IndexT, class SubsIterT> 
    void ind2sub_by_iter(const tensor_shape<T> &, const IndexT & ind, SubsIterT subs_iter) {}
    template <class SubsIterT, class IndexT, class T, class SizeT,  class ... SizeTs>
    void ind2sub_by_iter(const tensor_shape<T, SizeT, SizeTs...> & shape, const IndexT & ind, SubsIterT subs_iter) {
        const auto lm = shape.rest().magnitude();
        *subs_iter = ind / lm;
        ind2sub_by_iter(shape.rest(), ind % lm, ++subs_iter);
    }


    // invoke_with_subs
    template <class T, class IndexT, class FunT, class ... SubTs>
    constexpr decltype(auto) invoke_with_subs(const tensor_shape<T> & shape, const IndexT & ind, 
        FunT && fun, const SubTs & ... subs) {
        return forward<FunT>(fun)(subs ...);
    }
    template <class T, class IndexT, class FunT, class SizeT, class ... SizeTs, class ... SubTs>
    constexpr decltype(auto) invoke_with_subs(const tensor_shape<T, SizeT, SizeTs...> & shape, const IndexT & ind, 
        FunT && fun, const SubTs & ... subs) {
        return invoke_with_subs(shape.rest(), ind % shape.rest().magnitude(),
            forward<FunT>(fun), 
            subs ..., ind / shape.rest().magnitude());
    }




    // for_each_subscript
    template <class T, class FunT, class ... Ts>
    void for_each_subscript(const tensor_shape<T> &, FunT && fun, Ts && ... args) {
        fun(args ...);
    }
    template <class T, class SizeT, class ... SizeTs, class FunT, class ... Ts>
    void for_each_subscript(const tensor_shape<T, SizeT, SizeTs...> & shape, FunT && fun, Ts && ... args) {
        const auto n = shape.value();
        for (T i = 0; i < n; i++) {
            for_each_subscript(shape.rest(), fun, args ..., i);
        }
    }



    // for_each_subscript_if
    template <class T, class FunT, class ... Ts>
    constexpr bool for_each_subscript_if(const tensor_shape<T> &, FunT && fun, Ts && ... args) {
        return fun(args ...);
    }
    template <class T, class SizeT, class ... SizeTs, class FunT, class ... Ts>
    bool for_each_subscript_if(const tensor_shape<T, SizeT, SizeTs...> & shape, FunT && fun, Ts && ... args) {
        const auto n = shape.value();
        T i = 0;
        for (; i < n; i++) {
            if (!for_each_subscript_if(shape.rest(), fun, args ..., i))
                break;
        }
        return i == n;
    }




    // for_each_subscript_until
    template <class T, T Idx, class FunT, class ... Ts>
    std::enable_if_t<Idx == 0> for_each_subscript_until(const tensor_shape<T> &, 
        const const_ints<T, Idx> &, FunT && fun, Ts && ... args) {
        fun(args ...);
    }
    template <class T, class SizeT, class ... SizeTs, size_t Idx, class FunT, class ... Ts>
    std::enable_if_t<Idx == sizeof...(SizeTs)+1>
        for_each_subscript_until(const tensor_shape<T, SizeT, SizeTs...> & shape, const const_index<Idx> &, FunT && fun, Ts && ... args) {
        fun(args ...);
    }
    template <class T, class SizeT, class ... SizeTs, size_t Idx, class FunT, class ... Ts>
    std::enable_if_t<(Idx < sizeof...(SizeTs)+1)>
        for_each_subscript_until(const tensor_shape<T, SizeT, SizeTs...> & shape, const const_index<Idx> & idx, FunT && fun, Ts && ... args) {
        const auto n = value();
        for (T i = 0; i < n; i++) {
            for_each_subscript_until(shape.rest(), idx, fun, args ..., i);
        }
    }






    // ==
    template <class T, class K>
    constexpr bool operator == (const tensor_shape<T> &, const tensor_shape<K> &) { 
        return true; 
    }
    template <class T, class SizeT, class ... SizeTs, class K, class ... SizeT2s>
    constexpr bool operator == (const tensor_shape<T, SizeT, SizeTs...> & shape, const tensor_shape<K, SizeT2s...> & b) {
        return shape.value() == b.value() && shape.rest() == b.rest();
    }


    // !=
    template <class T1, class T2, class ... SizeT1s, class ... SizeT2s>
    constexpr bool operator != (const tensor_shape<T1, SizeT1s...> & s1, const tensor_shape<T2, SizeT2s...> & s2) {
        return !(s1 == s2);
    }



    // is_tensor_shape
    template <class T> 
    struct is_tensor_shape : no {};
    template <class T, class ... SizeTs>
    struct is_tensor_shape<tensor_shape<T, SizeTs...>> : yes {};



    namespace details {
        template <class T, class K, class = std::enable_if_t<std::is_integral<K>::value>>
        constexpr T _to_size(const K & s) {
            return s;
        }
        template <class T, class K, K Val>
        constexpr auto _to_size(const const_ints<K, Val> &) {
            return const_ints<T, Val>();
        }
    }

    // make_shape
    template <class T = size_t, class ... SizeTs>
    constexpr auto make_shape(const SizeTs & ... sizes) {
        return tensor_shape<T, decltype(details::_to_size<T>(sizes)) ...>(details::_to_size<T>(sizes) ...);
    }

    // make_shape
    template <class T = size_t, T ... Sizes>
    constexpr auto make_shape(const const_ints<T, Sizes...> &) {
        return tensor_shape<T, const_ints<T, Sizes>...>();
    }






    // stream
    namespace details {
        template <class ShapeT, size_t ... Is>
        inline std::ostream & _stream_seq(std::ostream & os, const ShapeT & shape, std::index_sequence<Is...>) {
            return print(" ", os << "[", shape.at(const_index<Is>()) ...) << "]";
        }
    }
    template <class T, class ... SizeTs>
    inline std::ostream & operator << (std::ostream & os, const tensor_shape<T, SizeTs...> & shape) {
        return details::_stream_seq(os, shape, std::make_index_sequence<sizeof...(SizeTs)>());
    }   


}