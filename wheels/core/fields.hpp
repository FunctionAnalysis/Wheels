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

    template <class T, class U, class V, class = std::enable_if_t<
        join_overloading<std::decay_t<T>, func_fields>::value>>
    constexpr decltype(auto) fields(T && t, U && usage, V && visitor) {
        return overloaded<func_fields, 
            info_for_overloading_t<std::decay_t<T>, func_fields>, 
            std::decay_t<U>, std::decay_t<V>
        >()(forward<T>(t), forward<U>(usage), forward<V>(visitor));
    }



    // tuple like types
    struct info_tuple_like {};
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



    // container types
    struct info_container {};
    template <class ContT, class VisitorT>
    class container_proxy {
    public:
        template <class C, class V>
        constexpr container_proxy(C && c, V && v) : _content(forward<C>(c)), _visitor(forward<V>(v)) {}
        constexpr container_proxy(const container_proxy &) = default;
        container_proxy(container_proxy &&) = default;
        container_proxy & operator = (const container_proxy &) = default;
        container_proxy & operator = (container_proxy &&) = default;
        template <class ContT2>
        container_proxy & operator = (const container_proxy<ContT2, VisitorT> & c) {
            _content = ContT(std::begin(c._content), std::end(c._content));
            _visitor = c._visitor;
            return *this;
        }

        const ContT & content() const { return _content; }
        ContT & content() { return _content; }
        const VisitorT & visitor() const { return _visitor; }
        VisitorT & visitor() { return _visitor; }

        decltype(auto) begin() const { return std::begin(_content); }
        decltype(auto) end() const { return std::end(_content); }
        decltype(auto) begin() { return std::begin(_content); }
        decltype(auto) end() { return std::end(_content); }

        auto size() const { return _content.size(); }
        decltype(auto) operator[](size_t i) const & { return _visitor.visit(_content[i]); }
        decltype(auto) operator[](size_t i) & { return _visitor.visit(_content[i]); }
    private:
        ContT _content;
        VisitorT _visitor;
    };

    template <class ContT1, class ContT2, class V>
    bool operator == (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return std::equal(c1.begin(), c1.end(), c2.begin(), c2.end(), [&c1, &c2](auto && e1, auto && e2) {
            return c1.visitor().visit(e1) == c2.visitor().visit(e2);
        });
    }
    template <class ContT1, class ContT2, class V>
    constexpr bool operator != (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return !(c1 == c2);
    }
    template <class ContT1, class ContT2, class V>
    bool operator < (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return std::lexicographical_compare(c1.begin(), c1.end(), c2.begin(), c2.end(), [&c1, &c2](auto && e1, auto && e2) {
            return c1.visitor().visit(e1) < c2.visitor().visit(e2);
        });
    }
    template <class ContT1, class ContT2, class V>
    bool operator > (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return c2 < c1;
    }
    template <class ContT1, class ContT2, class V>
    bool operator <= (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return !(c1 > c2);
    }
    template <class ContT1, class ContT2, class V>
    bool operator >= (const container_proxy<ContT1, V> & c1, const container_proxy<ContT2, V> & c2) {
        return !(c1 < c2);
    }


    template <class T> struct is_container_proxy : no {};
    template <class T, class V> struct is_container_proxy<container_proxy<T, V>> : yes {};
    template <class ContT, class VisitorT>
    constexpr decltype(auto) as_container(ContT && c, VisitorT && v) {
        return container_proxy<ContT, std::decay_t<VisitorT>>(forward<ContT>(c), forward<VisitorT>(v));
    }

    template <class U, class V>
    struct overloaded<func_fields, info_container, U, V> {
        template <class TT, class UU, class VV>
        constexpr decltype(auto) operator()(TT && t, UU &&, VV && v) const {
            return as_container(forward<TT>(t), forward<VV>(v));
        }
    };
    template <class T, class AllocT> struct info_for_overloading<std::vector<T, AllocT>, func_fields> { using type = info_container; };
    template <class T, class AllocT> struct info_for_overloading<std::list<T, AllocT>, func_fields> { using type = info_container; };
    template <class T, class AllocT> struct info_for_overloading<std::deque<T, AllocT>, func_fields> { using type = info_container; };



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

    // has_member_func_fields_simple
    namespace details {
        template <class T, class VisitorT>
        struct _has_member_func_fields_simple {
            template <class TT, class VV>
            static auto test(int) -> decltype(
                std::declval<TT>().fields(std::declval<VV>()),
                yes()) {
                return yes();
            }
            template <class, class>
            static no test(...) { return no(); }
            static const bool value = std::is_same<decltype(test<T, VisitorT>(1)), yes>::value;
        };
    }
    template <class T, class VisitorT>
    struct has_member_func_fields_simple : const_bool<details::_has_member_func_fields_simple<T, VisitorT>::value> {};


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

    // has_global_func_fields_simple
    namespace details {
        template <class T, class VisitorT>
        struct _has_global_func_fields_simple {
            template <class TT, class VV>
            static auto test(int) -> decltype(
                ::wheels::fields(std::declval<TT>(), std::declval<VV>()),
                yes()) {
                return yes();
            }
            template <class, class>
            static no test(...) { return no(); }
            static const bool value = std::is_same<decltype(test<T, VisitorT>(1)), yes>::value;
        };
    }
    template <class T, class VisitorT>
    struct has_global_func_fields_simple : const_bool<details::_has_global_func_fields_simple<T, VisitorT>::value> {};


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
        decltype(auto) visit(T && v) const {
            return forward<T>(v).fields(_usage, *this);
        }
        template <class T, wheels_distinguish_1, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            has_member_func_fields_simple<T, this_t>::value>>
        decltype(auto) visit(T && v) const {
            return forward<T>(v).fields(*this);
        }
        template <class T, wheels_distinguish_2, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            !has_member_func_fields_simple<T, this_t>::value &&
            has_global_func_fields<T, UsageT, this_t>::value >>
        decltype(auto) visit(T && v) const {
            return ::wheels::fields(forward<T>(v), _usage, *this);
        }
        template <class T, wheels_distinguish_3, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            !has_member_func_fields_simple<T, this_t>::value &&
            !has_global_func_fields<T, UsageT, this_t>::value &&
            has_global_func_fields_simple<T, this_t>::value >>
        decltype(auto) visit(T && v) const {
            return ::wheels::fields(forward<T>(v), *this);
        }
        template <class T, wheels_distinguish_4, class = std::enable_if_t<
            !has_member_func_fields<T, UsageT, this_t>::value &&
            !has_member_func_fields_simple<T, this_t>::value &&
            !has_global_func_fields<T, UsageT, this_t>::value &&
            !has_global_func_fields_simple<T, this_t>::value >>
        constexpr decltype(auto) visit(T && v) const {
            return static_cast<T &&>(v);
        }

        // pack all members
        template <class ... Ts>
        decltype(auto) operator()(Ts && ... vs) const {
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
        template <class ... ArgTs>
        constexpr decltype(auto) operator()(ArgTs && ... args) const {
            return std::tuple<ArgTs ...>(forward<ArgTs>(args) ...);
        }
    };
    struct visit_to_tuplize {};
    template <class T>
    auto tuplize(T && data) {
        auto visitor = make_field_visitor(pack_as_tuple(), visit_to_tuplize());
        return visitor.visit(forward<T>(data));
    }
    using tuplizer = field_visitor<pack_as_tuple, visit_to_tuplize>;



    namespace details {
        template <class T>
        struct _has_func_fields_to_tuplize {
            static constexpr bool value = 
                has_member_func_fields<T, visit_to_tuplize, tuplizer>::value ||
                has_member_func_fields_simple<T, tuplizer>::value ||
                has_global_func_fields<T, visit_to_tuplize, tuplizer>::value ||
                has_global_func_fields_simple<T, tuplizer>::value;
        };
    }

    // comparable
    template <class T>
    struct comparable {
        constexpr decltype(auto) as_tuple() const & {
            static_assert(details::_has_func_fields_to_tuplize<const T &>::value,
                "definition of fields(...) for const T & is required");
            using result_t = decltype(tuplize(static_cast<const T &>(*this)));
            static_assert(!std::is_same<T, result_t>::value, "tuplization failed");
            return tuplize(static_cast<const T &>(*this));
        }
        decltype(auto) as_tuple() & {
            static_assert(details::_has_func_fields_to_tuplize<T &>::value,
                "definition of fields(...) for T & is required");
            using result_t = decltype(tuplize(static_cast<T &>(*this)));
            static_assert(!std::is_same<T, result_t>::value, "tuplization failed");
            return tuplize(static_cast<T &>(*this));
        }
        constexpr decltype(auto) as_tuple() const && {
            static_assert(details::_has_func_fields_to_tuplize<const T &&>::value,
                "definition of fields(...) for const T && is required");
            using result_t = decltype(tuplize(static_cast<const T &&>(*this)));
            static_assert(!std::is_same<T, result_t>::value, "tuplization failed");
            return tuplize(static_cast<const T &&>(*this));
        }
        decltype(auto) as_tuple() && {
            static_assert(details::_has_func_fields_to_tuplize<T &&>::value,
                "definition of fields(...) for T && is required");
            using result_t = decltype(tuplize(static_cast<T &&>(*this)));
            static_assert(!std::is_same<T, result_t>::value, "tuplization failed");
            return tuplize(static_cast<T &&>(*this));
        }
    };

    template <class A, class B>
    constexpr bool operator == (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() == b.as_tuple();
    }
    template <class A, class B>
    constexpr bool operator != (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() != b.as_tuple();
    }
    template <class A, class B>
    constexpr bool operator < (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() < b.as_tuple();
    }
    template <class A, class B>
    constexpr bool operator <= (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() <= b.as_tuple();
    }
    template <class A, class B>
    constexpr bool operator > (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() > b.as_tuple();
    }
    template <class A, class B>
    constexpr bool operator >= (const comparable<A> & a, const comparable<B> & b) {
        return a.as_tuple() >= b.as_tuple();
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

}
