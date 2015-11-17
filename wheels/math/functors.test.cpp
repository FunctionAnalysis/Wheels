#include <gtest/gtest.h>
#include "functors.hpp"

using namespace wheels;
namespace con = concurrency;

TEST(math, platform_test) {

    con::accelerator default_acc;
    std::wcout << default_acc.device_path << "\n";
    std::wcout << default_acc.dedicated_memory << "\n";

    std::vector<con::accelerator> accs = con::accelerator::get_all();
    for (int i = 0; i < accs.size(); i++) {
        std::wcout << (accs[i].supports_cpu_shared_memory ?
            "CPU shared memory: true" : "CPU shared memory: false") << "\n";
        std::wcout << (accs[i].supports_double_precision ?
            "double precision: true" : "double precision: false") << "\n";
        std::wcout << (accs[i].supports_limited_double_precision ?
            "limited double precision: true" : "limited double precision: false") << "\n";
    }

    const int size = 50;

    std::vector<double> aCPP(size);
    std::vector<double> bCPP(size);
    std::vector<double> sumCPP(size);

    std::generate_n(aCPP.begin(), size, std::rand);
    std::generate_n(bCPP.begin(), size, std::rand);

    // Create C++ AMP objects.
    con::array_view<const double, 1> a(size, aCPP);
    con::array_view<const double, 1> b(size, bCPP);
    con::array_view<double, 1> sum(size, sumCPP);
    sum.discard_data();



    con::parallel_for_each(
        // Define the compute domain, which is the set of threads that are created.
        sum.extent,
        // Define the code to run on each thread on the accelerator.
        [=](con::index<1> idx) restrict(amp) {
        functor_plus<processor::gpu> plus;
        functor_minus<processor::gpu> minus;
        sum[idx] = minus(plus(a[idx], b[idx]), b[idx]);
    });


    // Print the results. The expected output is "7, 9, 11, 13, 15".
    for (int i = 0; i < size; i++) {
        std::cout << sum[i] << "\n";
    }

}

