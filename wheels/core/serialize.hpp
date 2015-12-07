#pragma once

#include <fstream>
#include <filesystem>

#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/forward_list.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/queue.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/stack.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>


#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>

#include "fields.hpp"

namespace wheels {

    // serializable
    struct visit_to_serialize {};

    template <class T>
    struct serializable {};

    template <class T, class ArcT, class = std::enable_if_t<
        has_member_func_fields<const T &, visit_to_serialize, ArcT>::value >>
    void save(ArcT & arc, const serializable<T> & data) {
        static_cast<const T &>(data).fields(visit_to_serialize(), arc);
    }
    //template <class T, class ArcT, class = std::enable_if_t<
    //    has_member_func_fields<T &, visit_to_serialize, ArcT>::value >>
    //void save(ArcT & arc, const serializable<T> & data, void * = nullptr) {
    //    const_cast<T &>(data).fields(visit_to_serialize(), arc);
    //}
    template <class T, class ArcT, class = std::enable_if_t<
        has_member_func_fields<T &, visit_to_serialize, ArcT>::value >>
    void load(ArcT & arc, serializable<T> & data) {
        static_cast<T &>(data).fields(visit_to_serialize(), arc);
    }

    template <class T, class ArcT, wheels_distinguish_1, class = std::enable_if_t<
        !has_member_func_fields<const T &, visit_to_serialize, ArcT>::value &&
        has_member_func_fields_simple<const T &, ArcT>::value >>
    void save(ArcT & arc, const serializable<T> & data) {
        static_cast<const T &>(data).fields(arc);
    }
    //template <class T, class ArcT, wheels_distinguish_1, class = std::enable_if_t<
    //    !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
    //    has_member_func_fields_simple<T &, ArcT>::value >>
    //void save(ArcT & arc, const serializable<T> & data, void * = nullptr) {
    //    const_cast<T &>(data).fields(arc);
    //}
    template <class T, class ArcT, wheels_distinguish_1, class = std::enable_if_t<
        !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
        has_member_func_fields_simple<T &, ArcT>::value >>
    void load(ArcT & arc, serializable<T> & data) {
        static_cast<T &>(data).fields(arc);
    }


    template <class T, class ArcT, wheels_distinguish_2, class = std::enable_if_t<
        !has_member_func_fields<const T &, visit_to_serialize, ArcT>::value &&
        !has_member_func_fields_simple<const T &, ArcT>::value &&
        has_global_func_fields<const T &, visit_to_serialize, ArcT>::value >>
    void save(ArcT & arc, const serializable<T> & data) {
        ::wheels::fields(static_cast<const T &>(data), visit_to_serialize(), arc);
    }
    //template <class T, class ArcT, wheels_distinguish_2, class = std::enable_if_t<
    //    !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
    //    !has_member_func_fields_simple<T &, ArcT>::value &&
    //    has_global_func_fields<T &, visit_to_serialize, ArcT>::value >>
    //void save(ArcT & arc, const serializable<T> & data, void * = nullptr) {
    //    ::wheels::fields(const_cast<T &>(data), visit_to_serialize(), arc);
    //}
    template <class T, class ArcT, wheels_distinguish_2, class = std::enable_if_t<
        !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
        !has_member_func_fields_simple<T &, ArcT>::value &&
        has_global_func_fields<T &, visit_to_serialize, ArcT>::value >>
    void load(ArcT & arc, serializable<T> & data) {
        ::wheels::fields(static_cast<T &>(data), visit_to_serialize(), arc);
    }


    template <class T, class ArcT, wheels_distinguish_3, class = std::enable_if_t<
        !has_member_func_fields<const T &, visit_to_serialize, ArcT>::value &&
        !has_member_func_fields_simple<const T &, ArcT>::value &&
        !has_global_func_fields<const T &, visit_to_serialize, ArcT>::value &&
        has_global_func_fields_simple<const T &, ArcT>::value >>
    void save(ArcT & arc, const serializable<T> & data) {
        ::wheels::fields(static_cast<const T &>(data), arc);
    }
    //template <class T, class ArcT, wheels_distinguish_3, class = std::enable_if_t<
    //    !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
    //    !has_member_func_fields_simple<T &, ArcT>::value &&
    //    !has_global_func_fields<T &, visit_to_serialize, ArcT>::value &&
    //    has_global_func_fields_simple<T &, ArcT>::value >>
    //void save(ArcT & arc, const serializable<T> & data, void * = nullptr) {
    //    ::wheels::fields(const_cast<T &>(data), arc);
    //}
    template <class T, class ArcT, wheels_distinguish_3, class = std::enable_if_t<
        !has_member_func_fields<T &, visit_to_serialize, ArcT>::value &&
        !has_member_func_fields_simple<T &, ArcT>::value &&
        !has_global_func_fields<T &, visit_to_serialize, ArcT>::value &&
        has_global_func_fields_simple<T &, ArcT>::value >>
    void load(ArcT & arc, serializable<T> & data) {
        ::wheels::fields(static_cast<T &>(data), arc);
    }

}