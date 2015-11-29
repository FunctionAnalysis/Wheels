#pragma once

#include <array>
#include <vector>
#include <map>

#include "tensor_frame.hpp"

namespace wheels {

    // category definitions


    // fix sized tensor based on std::array
    namespace ts_traits {

        // readable
        template <class ShapeT, class E, size_t N>
        struct readable_at_index<ts_category<ShapeT, std::array<E, N>>> : yes {};

        template <class ShapeT, class E, size_t N, class IndexT>
        constexpr const E & at_index_const_impl(const ts_category<ShapeT, std::array<E, N>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // writable
        template <class ShapeT, class E, size_t N>
        struct writable_at_index<ts_category<ShapeT, std::array<E, N>>> : yes {};

        template <class ShapeT, class E, size_t N, class IndexT>
        E & at_index_nonconst_impl(ts_category<ShapeT, std::array<E, N>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // const iteratable
        template <class ShapeT, class E, size_t N>
        struct const_iterator_type<ts_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::const_iterator;
        };

        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cbegin_impl(const ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cend_impl(const ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin() + N;
        }

        // nonconst iteratable
        template <class ShapeT, class E, size_t N>
        struct nonconst_iterator_type<ts_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::iterator;
        };

        template <class ShapeT, class E, size_t N>
        decltype(auto) begin_impl(ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        decltype(auto) end_impl(ts_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin() + N;
        }

    }
    
    template <class ShapeT, class E, size_t N>
    class ts_category<ShapeT, std::array<E, N>> 
        : public ts_category_base<ts_category<ShapeT, std::array<E, N>>> {
        using base_t = ts_category_base<ts_category<ShapeT, std::array<E, N>>>;
        static_assert(ShapeT::is_static, "");
    public:
        template <class ... EleTs>
        constexpr explicit ts_category(EleTs && ... eles)
            : base_t(ShapeT(), std::array<E, N>{{(E)forward<EleTs>(eles) ...}}) {}

        template <class CategoryT, bool RInd, bool RSub>
        constexpr ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) {
            assign_from(t);
        }

        constexpr ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };


    template <class E, size_t N>
    using vec_ = ts_category<tensor_shape<size_t, const_size<N>>, std::array<E, N>>;
    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;
    using vec4 = vec_<double, 4>;

    template <class E, size_t M, size_t N>
    using mat_ = ts_category<tensor_shape<size_t, const_size<M>, const_size<N>>, std::array<E, M * N>>;
    using mat2x2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3x2 = mat_<double, 3, 2>;
    using mat3x3 = mat_<double, 3, 3>;







    // dynamic sized tensor based on std::vector
    namespace ts_traits {

        // readable
        template <class ShapeT, class E, class AllocT>
        struct readable_at_index<ts_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class ShapeT, class E, class AllocT, class IndexT>
        decltype(auto) at_index_const_impl(const ts_category<ShapeT, std::vector<E, AllocT>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // writable
        template <class ShapeT, class E, class AllocT>
        struct writable_at_index<ts_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class ShapeT, class E, class AllocT, class IndexT>
        decltype(auto) at_index_nonconst_impl(ts_category<ShapeT, std::vector<E, AllocT>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // const iteratable
        template <class ShapeT, class E, class AllocT>
        struct const_iterator_type<ts_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::const_iterator;
        };

        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::const_iterator cbegin_impl(const ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::const_iterator cend_impl(const ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }

        // nonconst iteratable
        template <class ShapeT, class E, class AllocT>
        struct nonconst_iterator_type<ts_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::iterator;
        };

        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::iterator begin_impl(ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::iterator end_impl(ts_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }


        // reserve
        template <class ShapeT, class E, class AllocT>
        void reserve_impl(ts_category<ShapeT, std::vector<E, AllocT>> & t, size_t s) {
            t.data_provider().resize(s, 0);
        }

    }

    template <class ShapeT, class E, class AllocT>
    class ts_category<ShapeT, std::vector<E, AllocT>>
        : public ts_category_base<ts_category<ShapeT, std::vector<E, AllocT>>> {
        using base_t = ts_category_base<ts_category<ShapeT, std::vector<E, AllocT>>>;
    public:
        ts_category() : base_t(with_args, ShapeT(), ShapeT().magnitude()) {}
        template <class ... EleTs>
        explicit ts_category(const ShapeT & shape, EleTs && ... eles)
            : base_t(shape, std::vector<E, AllocT>({(E)forward<EleTs>(eles) ...})) {}

        template <class CategoryT, bool RInd, bool RSub>
        ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) : base_t() {
            assign_from(t);
        }

        ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };

    template <class ST, class E, class AllocT>
    class ts_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>
        : public ts_category_base<ts_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>> {
        using base_t = ts_category_base<ts_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>>;
    public:
        ts_category() : base_t(tensor_shape<ST, ST>(), std::vector<E, AllocT>()) {}
        template <class EleT, class ... EleTs>
        explicit ts_category(EleT && e, EleTs && ... eles) 
            : base_t(make_shape<ST>(1 + sizeof...(eles)), 
            std::vector<E, AllocT>({ (E)forward<EleT>(e), (E)forward<EleTs>(eles) ... })) {}

        template <class CategoryT, bool RInd, bool RSub>
        ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) : base_t() {
            assign_from(t);
        }

        ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };


    template <class E>
    using vecx_ = ts_category<tensor_shape<size_t, size_t>, std::vector<E>>;
    using vecx = vecx_<double>;

    template <class E>
    using matx_ = ts_category<tensor_shape<size_t, size_t, size_t>, std::vector<E>>;
    using matx = matx_<double>;








    // sparse tensor based on 
    template <class E, class IndexerT>
    struct sparse_dictionary {
        using value_type = E;
        static_assert(std::is_same<E, typename IndexerT::value_type::second_type>::value, "invalid IndexerT");
        using nonzero_iterator = second_in_pair_iterator_of<typename IndexerT::const_iterator>;

        IndexerT indexer;

        template <class IndexT>
        bool contains(const IndexT & ind) const {
            return indexer.find(ind) != indexer.end();
        }
        template <class IndexT>
        E at(const IndexT & ind, const E & default_val) const {
            return contains(ind) ? indexer.at(ind) : default_val;
        }
        template <class IndexT>
        E & at(const IndexT & ind) {
            return indexer[ind];
        }

        auto nzbegin() const {
            return nonzero_iterator(indexer.cbegin());
        }
        auto nzend() const {
            return nonzero_iterator(indexer.cend());
        }

        void clear() { indexer.clear(); }
    };

    namespace ts_traits {

        // readable
        template <class ShapeT, class E, class IndexerT>
        struct readable_at_index<ts_category<ShapeT, sparse_dictionary<E, IndexerT>>> : yes {};

        template <class ShapeT, class E, class IndexerT, class IndexT>
        E at_index_const_impl(const ts_category<ShapeT, sparse_dictionary<E, IndexerT>> & a, const IndexT & ind) {
            return a.data_provider().at(ind, 0);
        }

        // writable
        template <class ShapeT, class E, class IndexerT>
        struct writable_at_index<ts_category<ShapeT, sparse_dictionary<E, IndexerT>>> : yes {};

        template <class ShapeT, class E, class IndexerT, class IndexT>
        E & at_index_nonconst_impl(ts_category<ShapeT, sparse_dictionary<E, IndexerT>> & a, const IndexT & ind) {
            return a.data_provider().at(ind);
        }

        // nonzero iteratable
        template <class ShapeT, class E, class IndexerT>
        struct nonzero_iterator_type<ts_category<ShapeT, sparse_dictionary<E, IndexerT>>> {
            using type = typename sparse_dictionary<E, IndexerT>::nonzero_iterator;
        };

        template <class ShapeT, class E, class IndexerT>
        decltype(auto) nzbegin_impl(const ts_category<ShapeT, sparse_dictionary<E, IndexerT>> & a) {
            return a.data_provider().nzbegin();
        }
        template <class ShapeT, class E, class IndexerT>
        decltype(auto) nzend_impl(const ts_category<ShapeT, sparse_dictionary<E, IndexerT>> & a) {
            return a.data_provider().nzend();
        }


        // iterate all elements of 'from' 
        // since the 'from' does not have a nonzero iterator or 
        // index is inaccessible from the input nonzero iterator
        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, 
            class CategoryT2, bool RInd1, bool RInd2>
        void _assign_impl(ts_writable<ts_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const ts_readable<CategoryT2, RInd1, RInd2, true> & from, const no &){
            to.data_provider().clear();
            for (size_t ind = 0; ind < from.numel(); ind++) {
                auto e = from.at_index_const(ind);
                if (e) {
                    to.at_index_nonconst(ind) = e;
                }
            }
        }
        // use nonzero iterator of 'from'
        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, 
            class CategoryT2, class NZIterT2>
        void _assign_impl(ts_writable<ts_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const ts_nonzero_iteratable<CategoryT2, NZIterT2> & from, const yes &){
            to.data_provider().clear();
            for (auto it = from.nzbegin(); it != from.nzend(); ++it) {
                size_t ind = iter2ind(it);
                auto e = *it;
                to.at_index_nonconst(ind) = e;
            }
        }

        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, 
            class CategoryT2, bool RInd1, bool RInd2>
        void assign_impl(ts_writable<ts_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const ts_readable<CategoryT2, RInd1, RInd2, true> & from) {
            _assign_impl(to, from.category(),
                index_accessible_from_iterator<typename nonzero_iterator_type<CategoryT2>::type>());
        }
    }

    template <class ShapeT, class E, class IndexerT>
    class ts_category<ShapeT, sparse_dictionary<E, IndexerT>>
        : public ts_category_base<ts_category<ShapeT, sparse_dictionary<E, IndexerT>>> {
        using base_t = ts_category_base<ts_category<ShapeT, sparse_dictionary<E, IndexerT>>>;
    public:
        ts_category() {}
        template <class ... ElePairTs>
        explicit ts_category(const ShapeT & shape, ElePairTs && ... ele_pairs)
            : base_t(shape, sparse_dictionary<E, IndexerT>{{ele_pairs ...}}) {}

        template <class CategoryT, bool RInd, bool RSub>
        ts_category(const ts_readable<CategoryT, RInd, RSub, true> & t) {
            assign_from(t);
        }

        ts_category(const ts_category &) = default;
        ts_category(ts_category &&) = default;
        ts_category & operator = (const ts_category &) = default;
        ts_category & operator = (ts_category &&) = default;
    };


    template <class E>
    using spvec_ = ts_category<tensor_shape<size_t, size_t>, sparse_dictionary<E, std::map<size_t, E>>>;
    using spvec = spvec_<double>;

    template <class E>
    using spmat_ = ts_category<tensor_shape<size_t, size_t, size_t>, sparse_dictionary<E, std::map<size_t, E>>>;
    using spmat = spmat_<double>;


}