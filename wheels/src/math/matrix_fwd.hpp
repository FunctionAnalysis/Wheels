#pragma once

#include "../tensor/base_fwd.hpp"
#include "../tensor/extension_fwd.hpp"

namespace wheels {

struct extension_tag_matrix {};

template <class ET, class ST, class MT, class NT, class T>
class tensor_extension_base<extension_tag_matrix, ET, tensor_shape<ST, MT, NT>,
                            T>
    : public tensor_base<ET, tensor_shape<ST, MT, NT>, T> {
public:

};

template <class ET, class ST, class MT, class NT, class T>
using matrix_base = tensor_extension_base<extension_tag_matrix, ET,
                                          tensor_shape<ST, MT, NT>, T>;
}