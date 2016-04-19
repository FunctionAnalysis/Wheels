#pragma once

#include "../tensor/base.hpp"

#include "matrix_fwd.hpp"

namespace wheels {

template <class EleT, class ST, class MT, class NT, class T>
class matrix_base : public tensor_base<EleT, tensor_shape<ST, MT, NT>,
                                       matrix_base<EleT, ST, MT, NT, T>> {
public:

};
}
