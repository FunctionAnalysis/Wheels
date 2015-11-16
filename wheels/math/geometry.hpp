#pragma once

namespace wheels {
	
    struct vec_concept {
        // using value_type;
        // static constexpr size_t dimension;
        // value_type operator[](size_t i);
    };


    struct line_concept {
        // vec_concept first();
        // vec_concept second();
    };

    struct plane_concept {
        // vec_concept anchor();
        // vec_concept normal();
    };

    struct aligned_box_concept {
        // vec_concept min_corner();
        // vec_concept max_corner();
    };

}