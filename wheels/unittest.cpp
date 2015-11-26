#include <gtest/gtest.h>

int main(int argc, char ** argv) {
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_catch_exceptions = false;
    //testing::FLAGS_gtest_filter = "math.matrix";
    return RUN_ALL_TESTS();
}