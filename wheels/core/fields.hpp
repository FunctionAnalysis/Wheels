#pragma once

#include <tuple>
#include <vector>
#include <list>
#include <deque>
#include <array>
#include <utility>

#include "constants.hpp"
#include "types.hpp"
#include "overloads.hpp"

namespace wheels {

    // fields
    struct func_fields {};

    struct info_scalar {};
    struct info_container {};
    struct info_tuple_like {};

    template <class T, class U, class V, class = std::enable_if_t<
        join_overloading<std::decay_t<T>, func_fields>::value>>
    constexpr decltype(auto) fields(T && t, U && usage, V && visitor) {
        return overloaded<func_fields, 
            info_for_overloading_t<std::decay_t<T>, func_fields>, 
            std::decay_t<U>, std::decay_t<V>
        >()(forward<T>(t), forward<U>(usage), forward<V>(visitor));
    }


    // scalars
    //template <class T>
    //struct scalar_proxy {
    //    template <class TT>
    //    constexpr scalar_proxy(TT && c) : content(forward<TT>(c)) {}
    //    T content;
    //};
    //template <class T> struct is_scalar_proxy : no {};
    //template <class T> struct is_scalar_proxy<scalar_proxy<T>> : yes {};
    //template <class T>
    //constexpr auto as_scalar(T && c) {
    //    return scalar_proxy<T>(forward<T>(c));
    //}

    template <class U, class V>
    struct overloaded<func_fields, info_scalar, U, V> {
        template <class TT, class UU, class VV>
        constexpr decltype(auto) operator()(TT && t, UU &&, VV && visitor) const {
            return static_cast<TT&&>(t);//visitor(as_scalar(forward<TT>(t))); 
        }
    };
    template <> struct info_for_overloading<bool, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<char, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<uint8_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<uint16_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<uint32_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<uint64_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<int8_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<int16_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<int32_t, func_fields> { using type = info_scalar; };
    template <> struct info_for_overloading<int64_t, func_fields> { using type = info_scalar; };


    // containers
    template <class ContT>
    struct container_proxy {
        template <class C>
        constexpr container_proxy(C && c) : cont(forward<C>(c)) {}
        ContT content;
    };
    template <class T> struct is_container_proxy : no {};
    template <class T> struct is_container_proxy<container_proxy<T>> : yes {};
    template <class ContT>
    constexpr auto as_container(ContT && c) {
        return container_proxy<ContT>(forward<ContT>(c));
    }
    
    template <class U, class V>
    struct overloaded<func_fields, info_container, U, V> {
        template <class TT, class UU, class VV>
        constexpr decltype(auto) operator()(TT && t, UU &&, VV &&) const {
            return as_container(forward<TT>(t));
        }
    };
    template <class T, class AllocT> struct info_for_overloading<std::vector<T, AllocT>, func_fields> { using type = info_container; };
    template <class T, class AllocT> struct info_for_overloading<std::list<T, AllocT>, func_fields> { using type = info_container; };
    template <class T, class AllocT> struct info_for_overloading<std::deque<T, AllocT>, func_fields> { using type = info_container; };


    // tuple like types
    namespace details {
        template <class TupleT, class V, size_t ... Is>
        auto _fields_of_tuple_seq(TupleT && t, V && visitor, const const_ints<size_t, Is...> &) {
            return forward<V>(visitor)(std::get<Is>(forward<TupleT>(t)) ...);
        }
    }
    template <class U, class V>
    struct overloaded<func_fields, info_tuple_like, U, V> {
        template <class TT, class UU, class VV>
        constexpr decltype(auto) operator()(TT && t, UU &&, VV && visitor) const {
            return details::_fields_of_tuple_seq(forward<TT>(t), forward<VV>(visitor),
                make_const_sequence(const_size<std::tuple_size<std::decay_t<TT>>::value>()));
        }
    };
    template <class T1, class T2> struct info_for_overloading<std::pair<T1, T2>, func_fields> { using type = info_tuple_like; };
    template <class T, size_t N> struct info_for_overloading<std::array<T, N>, func_fields> { using type = info_tuple_like; };
    template <class ... Ts> struct info_for_overloading<std::tuple<Ts ...>, func_fields> { using type = info_tuple_like; };





    // has_member_func_fields
    namespace details {
        template <class T, class UsageT, class VisitorT>
        struct _has_member_func_fields {
            template <class TT, class UU, class VV>
            static auto test(int) -> decltype(
                std::declval<TT>().fields(std::declval<UU>(), std::declval<VV>()),
                yes()) {
                return yes();
            }
            template <class, class, class>
            static no test(...) { return no(); }
            static const bool value = std::is_same<decltype(test<T, UsageT, VisitorT>(1)), yes>::value;
        };
    }
    template <class T, class UsageT, class VisitorT>
    struct has_member_func_fields : const_bool<details::_has_member_func_fields<T, UsageT, VisitorT>::value> {};

    // has_global_func_fields
    namespace details {
        template <class T, class UsageT, class VisitorT>
        struct _has_global_func_fields {
            template <class TT, class UU, class VV>
            static auto test(int) -> decltype(
                ::wheels::fields(std::declval<TT>(), std::declval<UU>(), std::declval<VV>()),
                yes()) {
                return yes();
            }
            template <class, class, class>
            static no test(...) { return no(); }
            static const bool value = std::is_same<decltype(test<T, UsageT, VisitorT>(1)), yes>::value;
        };
    }
    template <class T, class UsageT, class VisitorT>
    struct has_global_func_fields : const_bool<details::_has_global_func_fields<T, UsageT, VisitorT>::value> {};



    // field_visitor
    template <class PackT, class UsageT>
    class field_visitor {
        using this_t = field_visitor<PackT, UsageT>;
    public:
        template <class PP, class UU>
        field_visitor(PP && p, UU && u) : _pack(forward<PP>(p)), _usage(forward<UU>(u)) {}

        // visit single member
        template <class T, class = std::enable_if_t<
            has_member_func_fields<T, UsageT, this_t>::value >>
        decltype(auto) visit(T && v) {
            return forward<T>(v).fields(_usage, *this);
        }
        template <class T, class = void, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            has_global_func_fields<T, UsageT, this_t>::value >>
        decltype(auto) visit(T && v) {
            return ::wheels::fields(forward<T>(v), _usage, *this);
        }
        template <class T, class = void, class = void, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            !has_global_func_fields<T, UsageT, this_t>::value >>
        constexpr T && visit(T && v) const {
            //static_assert(//is_scalar_proxy<std::decay_t<T>>::value || 
            //    is_container_proxy<std::decay_t<T>>::value,
            //    "no fields implementation is found");
            return static_cast<T&&>(v);
        }

        // pack all members
        template <class ... Ts>
        decltype(auto) operator()(Ts && ... vs) {
            return _pack(visit(forward<Ts>(vs))...);
        }

    private:
        PackT _pack;
        UsageT _usage;
    };
    
    // make_field_visitor
    template <class PP, class UU>
    constexpr auto make_field_visitor(PP && pack, UU && usage) {
        return field_visitor<std::decay_t<PP>, std::decay_t<UU>>(forward<PP>(pack), forward<UU>(usage));
    }



    // tuplize
    struct pack_as_tuple {
        template <class T>
        struct _each {
            template <class ArgT>
            static decltype(auto) process(ArgT && arg) {
                return static_cast<ArgT &&>(arg);
            }
        };
        template <class ... Ts>
        struct _each<std::tuple<Ts ...>> {
            template <class ArgT>
            static decltype(auto) process(ArgT && arg) {
                return forward<ArgT>(arg);
            }
        };
        template <class T>
        struct _each<container_proxy<T>> {
            template <class ArgT>
            static auto process(ArgT && arg) {

            }
        };
        template <class ... ArgTs>
        static decltype(auto) _pack(ArgTs && ... args) {
            return std::tuple<ArgTs ...>(forward<ArgTs>(args) ...);
        }
        template <class ... ArgTs>
        constexpr decltype(auto) operator()(ArgTs && ... args) const {
            return _pack(_each<std::decay_t<ArgTs>>::process(forward<ArgTs>(args)) ...);
        }
    };
    struct visit_to_tuplize {};
    template <class T>
    auto tuplize(T && data) {
        auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
        return visitor.visit(forward<T>(data));
    }
    //template <class T>
    //struct tuplizable {
    //    decltype(auto) tuplize() const & {
    //        auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
    //        return visitor.visit(static_cast<const T &>(*this));
    //    }
    //    decltype(auto) tuplize() & {
    //        auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
    //        return visitor.visit(static_cast<T &>(*this));
    //    }
    //};



    // serializable
    struct visit_to_serialize {};
    template <class T>
    struct serializable {
        template <class ArcT, class = std::enable_if_t<
            has_member_func_fields<T &, visit_to_serialize, ArcT>::value>>
        void serialize(ArcT & arc) {
            static_cast<T &>(*this).fields(visit_to_serialize(), arc);
        }
        template <class ArcT, class = void, class = std::enable_if_t<
            !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
            has_global_func_fields<T &, visit_to_serialize, ArcT>::value>>
        void serialize(ArcT & arc) {
            ::wheels::fields(static_cast<T &>(*this), visit_to_serialize(), arc);
        }
        template <class ArcT, class = void, class = void, class = std::enable_if_t<
            !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
            !has_global_func_fields<T &, visit_to_serialize, ArcT>::value >>
        void serialize(ArcT & arc) {
            static_assert(always<bool, false, T>::value, "no fields implementation is found, "
                "implement fields(...) or serialize(...) to fix this");
        }
    };

    // comparable
    template <class T>
    struct comparable {

    };


    //// object_comparable
    //template <class T>
    //struct object_comparable : object_traversing_fields<T> {};

    //template <class A, class B>
    //constexpr bool operator == (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() == b.as_tuple();
    //}
    //template <class A, class B>
    //constexpr bool operator != (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() != b.as_tuple();
    //}
    //template <class A, class B>
    //constexpr bool operator < (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() < b.as_tuple();
    //}
    //template <class A, class B>
    //constexpr bool operator <= (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() <= b.as_tuple();
    //}
    //template <class A, class B>
    //constexpr bool operator > (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() > b.as_tuple();
    //}
    //template <class A, class B>
    //constexpr bool operator >= (const object_comparable<A> & a, const object_comparable<B> & b) {
    //    return a.as_tuple() >= b.as_tuple();
    //}

}
