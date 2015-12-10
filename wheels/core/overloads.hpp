#pragma once

#include "constants.hpp"
#include "types.hpp"

namespace wheels {

// category forward declarations for overloading
struct fields_category_tuple_like {};
struct fields_category_container {};

struct category_const_expr {};

template <class ShapeT, class EleT, class T> struct category_tensor {};

// overload operators without losing any type information

// join_overloading
namespace details {
template <class T, class OpT> struct _join_overloading {
  template <class TT, class OpTT>
  static constexpr auto test(int)
      -> decltype(category_for_overloading(std::declval<const TT &>(),
                                           std::declval<const OpTT &>()),
                  yes()) {
    return yes();
  }
  template <class, class> static constexpr no test(...) { return no(); }
  static constexpr bool value =
      std::is_same<decltype(test<T, OpT>(1)), yes>::value;
};
}
template <class T, class OpT>
struct join_overloading
    : const_bool<details::_join_overloading<T, OpT>::value> {};

// category_for_overloading_t
namespace details {
template <class T, class OpT,
          bool JoinOverloading = _join_overloading<T, OpT>::value>
struct _category_for_overloading_helper {
  using type = void;
};
template <class T, class OpT>
struct _category_for_overloading_helper<T, OpT, true> {
  using type = decltype(category_for_overloading(std::declval<const T &>(),
                                                 std::declval<const OpT &>()));
};
}
template <class T, class OpT>
using category_for_overloading_t =
    typename details::_category_for_overloading_helper<T, OpT>::type;

// the overloaded<...> functor is called
// if any of the parameters join overloading
template <class OpT, class... ArgInfoTs> struct overloaded {
  constexpr overloaded() {}
  template <class... ArgTs> int operator()(ArgTs &&...) const {
    static_assert(
        always<bool, false, ArgTs...>::value,
        "error: this overloaded operator/function is not implemented, "
        "instantiate overloaded<...> to fix this.");
  }
};

// ewise
template <class OpT> struct func_ewise {
  constexpr func_ewise() : op() {}
  constexpr explicit func_ewise(const OpT &o) : op(o) {}
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... args) const {
    return op(forward<ArgTs>(args)...);
  }
  template <class ArcT> void serialize(ArcT &ar) { ar(op); }
  OpT op;
};

#define WHEELS_OVERLOAD_UNARY_OP(op, name)                                     \
  struct unary_op_##name {                                                     \
    constexpr unary_op_##name() {}                                             \
    template <class TT> constexpr decltype(auto) operator()(TT &&v) const {    \
      return (op forward<TT>(v));                                              \
    }                                                                          \
  };                                                                           \
  template <class T,                                                           \
            class =                                                            \
                std::enable_if_t<join_overloading<T, unary_op_##name>::value>> \
  constexpr decltype(auto) operator op(T &&v) {                                \
    return overloaded<unary_op_##name,                                         \
                      category_for_overloading_t<T, unary_op_##name>>()(       \
        forward<T>(v));                                                        \
  }

WHEELS_OVERLOAD_UNARY_OP(-, minus)
WHEELS_OVERLOAD_UNARY_OP(!, not)
WHEELS_OVERLOAD_UNARY_OP(~, bitwise_not)

#define WHEELS_OVERLOAD_BINARY_OP(op, name)                                    \
  struct binary_op_##name {                                                    \
    constexpr binary_op_##name() {}                                            \
    template <class TT1, class TT2>                                            \
    constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {            \
      return (forward<TT1>(v1) op forward<TT2>(v2));                           \
    }                                                                          \
  };                                                                           \
  template <class T1, class T2,                                                \
            class = std::enable_if_t<                                          \
                join_overloading<T1, binary_op_##name>::value ||               \
                join_overloading<T2, binary_op_##name>::value>>                \
  constexpr decltype(auto) operator op(T1 &&v1, T2 &&v2) {                     \
    return overloaded<binary_op_##name,                                        \
                      category_for_overloading_t<T1, binary_op_##name>,        \
                      category_for_overloading_t<T2, binary_op_##name>>()(     \
        forward<T1>(v1), forward<T2>(v2));                                     \
  }

WHEELS_OVERLOAD_BINARY_OP(+, plus)
WHEELS_OVERLOAD_BINARY_OP(-, minus)
WHEELS_OVERLOAD_BINARY_OP(*, mul)
WHEELS_OVERLOAD_BINARY_OP(/, div)
WHEELS_OVERLOAD_BINARY_OP(%, mod)

WHEELS_OVERLOAD_BINARY_OP(==, eq)
WHEELS_OVERLOAD_BINARY_OP(!=, neq)
WHEELS_OVERLOAD_BINARY_OP(<, lt)
WHEELS_OVERLOAD_BINARY_OP(<=, lte)
WHEELS_OVERLOAD_BINARY_OP(>, gt)
WHEELS_OVERLOAD_BINARY_OP(>=, gte)

WHEELS_OVERLOAD_BINARY_OP(&&, and)
WHEELS_OVERLOAD_BINARY_OP(||, or)
WHEELS_OVERLOAD_BINARY_OP(&, bitwise_and)
WHEELS_OVERLOAD_BINARY_OP(|, bitwise_or)
WHEELS_OVERLOAD_BINARY_OP (^, bitwise_xor)

#define WHEELS_OVERLOAD_UNARY_FUNC(name)                                       \
  struct func_##name {                                                         \
    constexpr func_##name() {}                                                 \
    template <class ArgT>                                                      \
    constexpr decltype(auto) operator()(ArgT &&v) const {                      \
      using std::name;                                                         \
      return name(forward<ArgT>(v));                                           \
    }                                                                          \
  };                                                                           \
  template <class T,                                                           \
            class = std::enable_if_t<join_overloading<T, func_##name>::value>, \
            class = void>                                                      \
  constexpr decltype(auto) name(T &&f) {                                       \
    return overloaded<func_##name,                                             \
                      category_for_overloading_t<T, func_##name>>()(           \
        forward<T>(f));                                                        \
  }

#define WHEELS_OVERLOAD_BINARY_FUNC(name)                                      \
  struct func_##name {                                                         \
    constexpr func_##name() {}                                                 \
    template <class ArgT1, class ArgT2>                                        \
    constexpr decltype(auto) operator()(ArgT1 &&v1, ArgT2 &&v2) const {        \
      using std::name;                                                         \
      return name(forward<ArgT1>(v1), forward<ArgT2>(v2));                     \
    }                                                                          \
  };                                                                           \
  template <                                                                   \
      class T1, class T2,                                                      \
      class = std::enable_if_t<any(join_overloading<T1, func_##name>::value,   \
                                   join_overloading<T2, func_##name>::value)>, \
      class = void>                                                            \
  constexpr decltype(auto) name(T1 &&t1, T2 &&t2) {                            \
    return overloaded<func_##name,                                             \
                      category_for_overloading_t<T1, func_##name>,             \
                      category_for_overloading_t<T2, func_##name>>()(          \
        forward<T1>(t1), forward<T2>(t2));                                     \
  }

#define WHEELS_OVERLOAD_FUNC(name)                                             \
  struct func_##name {                                                         \
    constexpr func_##name() {}                                                 \
    template <class... ArgTs>                                                  \
    constexpr decltype(auto) operator()(ArgTs &&... vs) const {                \
      using std::name;                                                         \
      return name(forward<ArgTs>(vs)...);                                      \
    }                                                                          \
  };                                                                           \
  template <class FirstT, class... RestTs,                                     \
            class = std::enable_if_t<any(                                      \
                join_overloading<FirstT, func_##name>::value,                  \
                join_overloading<RestTs, func_##name>::value...)>,             \
            class = void>                                                      \
  constexpr decltype(auto) name(FirstT &&f, RestTs &&... rests) {              \
    return overloaded<func_##name,                                             \
                      category_for_overloading_t<FirstT, func_##name>,         \
                      category_for_overloading_t<RestTs, func_##name>...>()(   \
        forward<FirstT>(f), forward<RestTs>(rests)...);                        \
  }

WHEELS_OVERLOAD_UNARY_FUNC(sin)
WHEELS_OVERLOAD_UNARY_FUNC(sinh)
WHEELS_OVERLOAD_UNARY_FUNC(asin)
WHEELS_OVERLOAD_UNARY_FUNC(asinh)
WHEELS_OVERLOAD_UNARY_FUNC(cos)
WHEELS_OVERLOAD_UNARY_FUNC(cosh)
WHEELS_OVERLOAD_UNARY_FUNC(acos)
WHEELS_OVERLOAD_UNARY_FUNC(acosh)
WHEELS_OVERLOAD_UNARY_FUNC(tan)
WHEELS_OVERLOAD_UNARY_FUNC(tanh)
WHEELS_OVERLOAD_UNARY_FUNC(atan)
WHEELS_OVERLOAD_UNARY_FUNC(atanh)
WHEELS_OVERLOAD_UNARY_FUNC(log)
WHEELS_OVERLOAD_UNARY_FUNC(log2)
WHEELS_OVERLOAD_UNARY_FUNC(log10)
WHEELS_OVERLOAD_UNARY_FUNC(exp)
WHEELS_OVERLOAD_UNARY_FUNC(exp2)
WHEELS_OVERLOAD_UNARY_FUNC(ceil)
WHEELS_OVERLOAD_UNARY_FUNC(floor)
WHEELS_OVERLOAD_UNARY_FUNC(round)
WHEELS_OVERLOAD_UNARY_FUNC(isinf)
WHEELS_OVERLOAD_UNARY_FUNC(isfinite)
WHEELS_OVERLOAD_UNARY_FUNC(isnan)

WHEELS_OVERLOAD_BINARY_FUNC(atan2)
WHEELS_OVERLOAD_BINARY_FUNC(pow)
WHEELS_OVERLOAD_BINARY_FUNC(min)
WHEELS_OVERLOAD_BINARY_FUNC(max)

// object_overloading
template <class DerivedT, class OpT> struct object_overloading {};
template <class DerivedT, class... OpTs>
struct object_overloadings : object_overloading<DerivedT, OpTs>... {};

#define WHEELS_OVERLOAD_MEMBER_UNARY_OP(op1, op2, op3, opsymbol, name)         \
  struct member_op_##name {                                                    \
    constexpr member_op_##name() {}                                            \
    template <class CallerT, class ArgT>                                       \
    constexpr decltype(auto) operator()(CallerT &&caller, ArgT &&arg) const {  \
      return op1 forward<CallerT>(caller) op2 forward<ArgT>(arg) op3;          \
    }                                                                          \
  };                                                                           \
  template <class DerivedT>                                                    \
  struct object_overloading<DerivedT, member_op_##name> {                      \
    template <class ArgT>                                                      \
    constexpr decltype(auto) operator opsymbol(ArgT &&arg) const & {           \
      return overloaded<                                                       \
          member_op_##name, DerivedT,                                          \
          category_for_overloading_t<std::decay_t<ArgT>, member_op_##name>>()( \
          static_cast<const DerivedT &>(*this), forward<ArgT>(arg));           \
    }                                                                          \
    template <class ArgT> decltype(auto) operator opsymbol(ArgT &&arg) & {     \
      return overloaded<                                                       \
          member_op_##name, DerivedT,                                          \
          category_for_overloading_t<std::decay_t<ArgT>, member_op_##name>>()( \
          static_cast<DerivedT &>(*this), forward<ArgT>(arg));                 \
    }                                                                          \
    template <class ArgT> decltype(auto) operator opsymbol(ArgT &&arg) && {    \
      return overloaded<                                                       \
          member_op_##name, DerivedT,                                          \
          category_for_overloading_t<std::decay_t<ArgT>, member_op_##name>>()( \
          static_cast<DerivedT &&>(*this), forward<ArgT>(arg));                \
    }                                                                          \
    template <class ArgT>                                                      \
    decltype(auto) operator opsymbol(ArgT &&arg) const && {                    \
      return overloaded<                                                       \
          member_op_##name, DerivedT,                                          \
          category_for_overloading_t<std::decay_t<ArgT>, member_op_##name>>()( \
          static_cast<const DerivedT &&>(*this), forward<ArgT>(arg));          \
    }                                                                          \
  };

#define WHEELS_SYMBOL_LEFT_BRACKET [
#define WHEELS_SYMBOL_RIGHT_BRACKET ]
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, WHEELS_SYMBOL_LEFT_BRACKET,
                                WHEELS_SYMBOL_RIGHT_BRACKET, [], bracket)
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, +=, , +=, plus_equal)
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, -=, , -=, minus_equal)
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, *=, , *=, mul_equal)
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, /=, , /=, div_equal)
WHEELS_OVERLOAD_MEMBER_UNARY_OP(, =, , =, assign)

#define WHEELS_OVERLOAD_MEMBER_VARARG_OP(op1, op2, op3, opsymbol, name)        \
  struct member_op_##name {                                                    \
    constexpr member_op_##name() {}                                            \
    template <class CallerT, class... ArgTs>                                   \
    constexpr decltype(auto) operator()(CallerT &&caller,                      \
                                        ArgTs &&... args) const {              \
      return op1 forward<CallerT>(caller) op2 forward<ArgTs>(args)... op3;     \
    }                                                                          \
  };                                                                           \
  template <class DerivedT>                                                    \
  struct object_overloading<DerivedT, member_op_##name> {                      \
    template <class... ArgTs>                                                  \
    constexpr decltype(auto) operator opsymbol(ArgTs &&... args) const & {     \
      return overloaded<member_op_##name, DerivedT,                            \
                        category_for_overloading_t<std::decay_t<ArgTs>,        \
                                                   member_op_##name>...>()(    \
          static_cast<const DerivedT &>(*this), forward<ArgTs>(args)...);      \
    }                                                                          \
    template <class... ArgTs>                                                  \
    decltype(auto) operator opsymbol(ArgTs &&... args) & {                     \
      return overloaded<member_op_##name, DerivedT,                            \
                        category_for_overloading_t<std::decay_t<ArgTs>,        \
                                                   member_op_##name>...>()(    \
          static_cast<DerivedT &>(*this), forward<ArgTs>(args)...);            \
    }                                                                          \
    template <class... ArgTs>                                                  \
    decltype(auto) operator opsymbol(ArgTs &&... args) && {                    \
      return overloaded<member_op_##name, DerivedT,                            \
                        category_for_overloading_t<std::decay_t<ArgTs>,        \
                                                   member_op_##name>...>()(    \
          static_cast<DerivedT &&>(*this), forward<ArgTs>(args)...);           \
    }                                                                          \
    template <class... ArgTs>                                                  \
    decltype(auto) operator opsymbol(ArgTs &&... args) const && {              \
      return overloaded<member_op_##name, DerivedT,                            \
                        category_for_overloading_t<std::decay_t<ArgTs>,        \
                                                   member_op_##name>...>()(    \
          static_cast<const DerivedT &&>(*this), forward<ArgTs>(args)...);     \
    }                                                                          \
  };

#define WHEELS_SYMBOL_LEFT_PAREN (
#define WHEELS_SYMBOL_RIGHT_PAREN )
WHEELS_OVERLOAD_MEMBER_VARARG_OP(, WHEELS_SYMBOL_LEFT_PAREN,
                                 WHEELS_SYMBOL_RIGHT_PAREN, (), paren)
}