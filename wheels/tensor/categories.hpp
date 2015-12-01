#pragma once

#include <array>
#include <vector>
#include <map>

#include "structure.hpp"

namespace wheels {

    // category definitions
    // std::array
    namespace tensor_traits {

        // readable
        template <class ShapeT, class E, size_t N>
        struct readable_at_index<tensor_category<ShapeT, std::array<E, N>>> : yes {};

        template <class ShapeT, class E, size_t N, class IndexT>
        constexpr const E & at_index_const_impl(const tensor_category<ShapeT, std::array<E, N>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // writable
        template <class ShapeT, class E, size_t N>
        struct writable_at_index<tensor_category<ShapeT, std::array<E, N>>> : yes {};

        template <class ShapeT, class E, size_t N, class IndexT>
        E & at_index_nonconst_impl(tensor_category<ShapeT, std::array<E, N>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // const iteratable
        template <class ShapeT, class E, size_t N>
        struct const_iterator_type<tensor_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::const_iterator;
        };

        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cbegin_impl(const tensor_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        constexpr decltype(auto) cend_impl(const tensor_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin() + N;
        }

        // nonconst iteratable
        template <class ShapeT, class E, size_t N>
        struct nonconst_iterator_type<tensor_category<ShapeT, std::array<E, N>>> {
            using type = typename std::array<E, N>::iterator;
        };

        template <class ShapeT, class E, size_t N>
        decltype(auto) begin_impl(tensor_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, size_t N>
        decltype(auto) end_impl(tensor_category<ShapeT, std::array<E, N>> & a) {
            return a.data_provider().begin() + N;
        }

    }
    
    template <class ShapeT, class E, size_t N>
    class tensor_category<ShapeT, std::array<E, N>> 
        : public tensor_category_base<tensor_category<ShapeT, std::array<E, N>>> {

        using base_t = tensor_category_base<tensor_category<ShapeT, std::array<E, N>>>;
        static_assert(ShapeT::is_static, "");
    public:
        using shape_type = ShapeT;
        using data_provider_type = std::array<E, N>;
        using value_type = E;
        static constexpr size_t rank = ShapeT::rank;

        template <class ... EleTs>
        constexpr explicit tensor_category(EleTs && ... eles)
            : base_t(ShapeT(), std::array<E, N>{{(E)forward<EleTs>(eles) ...}}) {}

        WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION
    };


    template <class E, size_t N>
    using vec_ = tensor_category<tensor_shape<size_t, const_size<N>>, std::array<E, N>>;
    using vec2 = vec_<double, 2>;
    using vec3 = vec_<double, 3>;
    using vec4 = vec_<double, 4>;

    template <class E, size_t M, size_t N>
    using mat_ = tensor_category<tensor_shape<size_t, const_size<M>, const_size<N>>, std::array<E, M * N>>;
    using mat2x2 = mat_<double, 2, 2>;
    using mat2x3 = mat_<double, 2, 3>;
    using mat3x2 = mat_<double, 3, 2>;
    using mat3x3 = mat_<double, 3, 3>;







    // std::vector
    namespace tensor_traits {

        // readable
        template <class ShapeT, class E, class AllocT>
        struct readable_at_index<tensor_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class ShapeT, class E, class AllocT, class IndexT>
        decltype(auto) at_index_const_impl(const tensor_category<ShapeT, std::vector<E, AllocT>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // writable
        template <class ShapeT, class E, class AllocT>
        struct writable_at_index<tensor_category<ShapeT, std::vector<E, AllocT>>> : yes {};

        template <class ShapeT, class E, class AllocT, class IndexT>
        decltype(auto) at_index_nonconst_impl(tensor_category<ShapeT, std::vector<E, AllocT>> & a, const IndexT & ind) {
            return a.data_provider()[ind];
        }

        // const iteratable
        template <class ShapeT, class E, class AllocT>
        struct const_iterator_type<tensor_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::const_iterator;
        };

        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::const_iterator cbegin_impl(const tensor_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::const_iterator cend_impl(const tensor_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }

        // nonconst iteratable
        template <class ShapeT, class E, class AllocT>
        struct nonconst_iterator_type<tensor_category<ShapeT, std::vector<E, AllocT>>> {
            using type = typename std::vector<E, AllocT>::iterator;
        };

        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::iterator begin_impl(tensor_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().begin();
        }
        template <class ShapeT, class E, class AllocT>
        typename std::vector<E, AllocT>::iterator end_impl(tensor_category<ShapeT, std::vector<E, AllocT>> & a) {
            return a.data_provider().end();
        }


        // reserve
        template <class ShapeT, class E, class AllocT>
        void reserve_impl(tensor_category<ShapeT, std::vector<E, AllocT>> & t, size_t s) {
            t.data_provider().resize(s, 0);
        }

    }

    template <class ShapeT, class E, class AllocT>
    class tensor_category<ShapeT, std::vector<E, AllocT>>
        : public tensor_category_base<tensor_category<ShapeT, std::vector<E, AllocT>>> {
        using base_t = tensor_category_base<tensor_category<ShapeT, std::vector<E, AllocT>>>;
    public:
        using shape_type = ShapeT;
        using data_provider_type = std::vector<E, AllocT>;
        using value_type = E;
        static constexpr size_t rank = ShapeT::rank;

        tensor_category() : base_t(with_args, ShapeT(), ShapeT().magnitude()) {}
        template <class ... EleTs>
        explicit tensor_category(const ShapeT & shape, EleTs && ... eles)
            : base_t(shape, std::vector<E, AllocT>({(E)forward<EleTs>(eles) ...})) {
            data_provider().resize(numel());
        }

        WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION
    };

    template <class ST, class E, class AllocT>
    class tensor_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>
        : public tensor_category_base<tensor_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>> {
        using base_t = tensor_category_base<tensor_category<tensor_shape<ST, ST>, std::vector<E, AllocT>>>;
    public:
        using shape_type = tensor_shape<ST, ST>;
        using data_provider_type = std::vector<E, AllocT>;
        using value_type = E;
        static constexpr size_t rank = 1;

        tensor_category() : base_t(tensor_shape<ST, ST>(), std::vector<E, AllocT>()) {}
        template <class EleT, class ... EleTs>
        explicit tensor_category(EleT && e, EleTs && ... eles) 
            : base_t(make_shape<ST>(1 + sizeof...(eles)), 
            std::vector<E, AllocT>({ (E)forward<EleT>(e), (E)forward<EleTs>(eles) ... })) {
            data_provider().resize(numel());
        }

        WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION
    };


    template <class E>
    using vecx_ = tensor_category<tensor_shape<size_t, size_t>, std::vector<E>>;
    using vecx = vecx_<double>;

    template <class E>
    using matx_ = tensor_category<tensor_shape<size_t, size_t, size_t>, std::vector<E>>;
    using matx = matx_<double>;








    // sparse tensor based on 
    template <class E, class IndexerT = std::map<size_t, E>>
    class sparse_dictionary {
        static_assert(std::is_same<E, typename IndexerT::value_type::second_type>::value, 
            "invalid IndexerT value type");
    public:
        using key_type = typename IndexerT::value_type::first_type;
        using value_type = E;
        using key_value_pair_type = std::pair<key_type, value_type>;
        using nonzero_iterator = nonzero_iterator_wrapper<
            pair_second_iterator_wrapper<typename IndexerT::const_iterator>
        >;

    public:
        constexpr sparse_dictionary() {}
        constexpr sparse_dictionary(const IndexerT & ind) : _indexer(ind) {
            clear_zeros();
        }
        constexpr sparse_dictionary(IndexerT && ind) : _indexer(std::move(ind)) {}
        constexpr sparse_dictionary(std::initializer_list<key_value_pair_type> ilist)
            : _indexer(ilist) {
            clear_zeros();
        }

    public:
        constexpr auto size() const { return _indexer.size(); }
        template <class IndexT>
        constexpr bool contains(const IndexT & ind) const {
            return _indexer.find(ind) != _indexer.end();
        }
        template <class IndexT>
        constexpr E at(const IndexT & ind, const E & default_val) const {
            return contains(ind) ? _indexer.at(ind) : default_val;
        }
        template <class IndexT>
        E & at(const IndexT & ind) {
            return _indexer[ind];
        }

        constexpr nonzero_iterator nzbegin() const {
            return wrap_nonzero_iterator(
                wrap_pair_second_iterator(_indexer.cbegin()),
                wrap_pair_second_iterator(_indexer.cend()));
        }
        constexpr nonzero_iterator nzend() const {
            return wrap_nonzero_iterator(
                wrap_pair_second_iterator(_indexer.cend()),
                wrap_pair_second_iterator(_indexer.cend()));
        }

        void clear_zeros() {
            auto it = _indexer.begin();
            while (it != _indexer.end()) {
                if (it->second == types<value_type>::zero()) {
                    it = _indexer.erase(it);
                } else {
                    ++it;
                }
            }
        }
        void clear() { _indexer.clear(); }

    private:
        IndexerT _indexer;
    };

    namespace tensor_traits {

        // readable
        template <class ShapeT, class E, class IndexerT>
        struct readable_at_index<tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>> : yes {};

        template <class ShapeT, class E, class IndexerT, class IndexT>
        E at_index_const_impl(const tensor_category<ShapeT, sparse_dictionary<E, IndexerT>> & a, const IndexT & ind) {
            return a.data_provider().at(ind, 0);
        }

        // writable
        template <class ShapeT, class E, class IndexerT>
        struct writable_at_index<tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>> : yes {};

        template <class ShapeT, class E, class IndexerT, class IndexT>
        E & at_index_nonconst_impl(tensor_category<ShapeT, sparse_dictionary<E, IndexerT>> & a, const IndexT & ind) {
            return a.data_provider().at(ind);
        }

        // nonzero iteratable
        template <class ShapeT, class E, class IndexerT>
        struct nonzero_iterator_type<tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>> {
            using type = typename sparse_dictionary<E, IndexerT>::nonzero_iterator;
        };

        template <class ShapeT, class E, class IndexerT>
        decltype(auto) nzbegin_impl(const tensor_category<ShapeT, sparse_dictionary<E, IndexerT>> & a) {
            return a.data_provider().nzbegin();
        }
        template <class ShapeT, class E, class IndexerT>
        decltype(auto) nzend_impl(const tensor_category<ShapeT, sparse_dictionary<E, IndexerT>> & a) {
            return a.data_provider().nzend();
        }

        // iterate all elements of 'from' 
        // since the 'from' does not have a nonzero iterator or 
        // index is inaccessible from the input nonzero iterator
        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, class CategoryT2, bool RInd1, bool RInd2>
        void _assign_impl(tensor_writable<tensor_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const tensor_readable<CategoryT2, RInd1, RInd2, true> & from, const no &) {
            to.data_provider().clear();
            for (size_t ind = 0; ind < from.numel(); ind++) {
                auto e = from.at_index_const(ind);
                if (e) {
                    to.at_index_nonconst(ind) = e;
                }
            }
        }
        // use nonzero iterator of 'from'
        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, class CategoryT2, class NZIterT2>
        void _assign_impl(tensor_writable<tensor_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const tensor_nonzero_iteratable<CategoryT2, NZIterT2> & from, const yes &) {
            to.data_provider().clear();
            for (auto it = from.nzbegin(); it != from.nzend(); ++it) {
                size_t ind = tensor_traits::iter2ind(it);
                auto e = *it;
                to.at_index_nonconst(ind) = e;
            }
        }

        template <class ShapeT1, class E, class IndexerT, bool WInd1, bool WInd2, class CategoryT2, bool RInd1, bool RInd2>
        void assign_impl(tensor_writable<tensor_category<ShapeT1, sparse_dictionary<E, IndexerT>>, WInd1, WInd2, true> & to,
            const tensor_readable<CategoryT2, RInd1, RInd2, true> & from) {
            _assign_impl(to, from.category(),
                index_accessible_from_iterator<typename nonzero_iterator_type<CategoryT2>::type>());
        }
    }

    template <class ShapeT, class E, class IndexerT>
    class tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>
        : public tensor_category_base<tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>> {
        using base_t = tensor_category_base<tensor_category<ShapeT, sparse_dictionary<E, IndexerT>>>;
        using key_value_pair_t = typename sparse_dictionary<E, IndexerT>::key_value_pair_type;
    public:
        using shape_type = ShapeT;
        using data_provider_type = sparse_dictionary<E, IndexerT>;
        using value_type = E;
        static constexpr size_t rank = ShapeT::rank;

        tensor_category() {}
        explicit tensor_category(const ShapeT & shape) 
            : base_t(shape, sparse_dictionary<E, IndexerT>()) {}
        explicit tensor_category(const ShapeT & shape, std::initializer_list<key_value_pair_t> ilist)
            : base_t(shape, sparse_dictionary<E, IndexerT>(ilist)) {}

        WHEELS_TENSOR_CATEGORY_COMMON_DEFINITION
    };


    template <class E>
    using spvec_ = tensor_category<tensor_shape<size_t, size_t>, sparse_dictionary<E, std::map<size_t, E>>>;
    using spvec = spvec_<double>;

    template <class E>
    using spmat_ = tensor_category<tensor_shape<size_t, size_t, size_t>, sparse_dictionary<E, std::map<size_t, E>>>;
    using spmat = spmat_<double>;








    // tensor_category_default
    template <class E, class ShapeT, bool Dense, 
        bool StaticShape = ShapeT::is_static>
    struct tensor_category_default {
        using type = tensor_category<ShapeT, std::vector<E>>;
    };
    template <class E, class ShapeT>
    struct tensor_category_default<E, ShapeT, true, true> {
        using type = tensor_category<ShapeT, std::array<E, ShapeT::static_magnitude>>;
    };
    template <class E, class ShapeT, bool StaticShape>
    struct tensor_category_default<E, ShapeT, false, StaticShape> {
        using type = tensor_category<ShapeT, sparse_dictionary<E, std::map<size_t, E>>>;
    };

    // eval
    template <bool Dense = true, class ShapeT, class DataProviderT, bool RInd, bool RSub>
    constexpr auto eval(const tensor_readable<tensor_category<ShapeT, DataProviderT>, RInd, RSub, true> & t) {
        using value_t = typename tensor_category<ShapeT, DataProviderT>::value_type;
        using eval_t = typename tensor_category_default<value_t, ShapeT, Dense>::type;
        return eval_t(t);
    }


}