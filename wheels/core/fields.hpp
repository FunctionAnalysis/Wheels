#pragma once

#include <tuple>

#include "constants.hpp"
#include "types.hpp"

namespace wheels {


    // fields for fundamental types and containers
    template <class T, class U, class V, class = std::enable_if_t<
        std::is_fundamental<std::decay_t<T>>::value>>
    constexpr T && fields(T && t, U &&, V &&) {
        return static_cast<T&&>(t);
    }

    template <class T, class AllocT, class U, class V>
    auto fields(std::vector<T, AllocT> & t, U && usage, V && visitor) {
        // todo           
    }




    // has_member_func_fields
    namespace details {
        template <class T, class UsageT, class VisitorT>
        struct _has_member_func_fields {
            template <class TT, class UU, class VV>
            static auto test(int) -> decltype(
                std::declval<TT &>().fields(std::declval<UU>(), std::declval<VV>()),
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
                ::wheels::fields(std::declval<TT &>(), std::declval<UU>(), std::declval<VV>()),
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

        // pack all members
        template <class ... Ts>
        decltype(auto) operator()(Ts && ... vs) {
            return _pack(visit(forward<Ts>(vs))...);
        }

        // visit single member
        template <class T, class = std::enable_if_t<
            has_member_func_fields<T, UsageT, this_t>::value>>
        decltype(auto) visit(T && v) {
            return forward<T>(v).fields(_usage, *this);
        }
        template <class T, class = void, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            has_global_func_fields<T, UsageT, this_t>::value>>
        decltype(auto) visit(T && v) {
            return ::wheels::fields(forward<T>(v), _usage, *this);
        }
        template <class T, class = void, class = void, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            !has_global_func_fields<T, UsageT, this_t>::value >>
        decltype(auto) visit(T && v) {
            static_assert(always<bool, false, T>::value, "no fields implementation is found");
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
        template <class ... ArgTs>
        constexpr std::tuple<ArgTs ...> operator()(ArgTs && ... args) const {
            return std::tuple<ArgTs ...>(forward<ArgTs>(args) ...);
        }
    };
    struct visit_to_tuplize {};
    template <class T>
    decltype(auto) tuplize(T && data) {
        auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
        return visitor.visit(forward<T>(data));
    }



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
