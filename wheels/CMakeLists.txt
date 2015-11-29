set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# all files
file (GLOB core_sources 
	"${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu" 
	"${CMAKE_CURRENT_SOURCE_DIR}/core/*.cuh"
)
source_group ("core" FILES ${core_sources})

file (GLOB tensor_sources 
	"${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.cu" 
	"${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.cuh")
source_group ("tensor" FILES ${tensor_sources})

file (GLOB math_sources 
	"${CMAKE_CURRENT_SOURCE_DIR}/math/*.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/math/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/math/*.cu" 
	"${CMAKE_CURRENT_SOURCE_DIR}/math/*.cuh")
source_group ("math" FILES ${math_sources})

file (GLOB feature_sources 
	"${CMAKE_CURRENT_SOURCE_DIR}/feature/*.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/feature/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/feature/*.cu" 
	"${CMAKE_CURRENT_SOURCE_DIR}/feature/*.cuh")
source_group ("feature" FILES ${feature_sources})


set (wheels_sources 
    ${core_sources} 
    ${tensor_sources} 
    ${math_sources} 
    ${feature_sources})
file (GLOB_RECURSE wheels_test_sources "." *.test.cpp *.test.hpp *.test.cu *.test.cuh)
# remove test sources from the lib sources
list (REMOVE_ITEM wheels_sources ${wheels_test_sources})


# the lib project
message(STATUS "wheels_sources:")
foreach(i ${wheels_sources})
	message (STATUS ${i})  
endforeach()
wheels_add_library(Wheels ${wheels_sources} ./dummy.cpp)




# the test project
message(STATUS "wheels_test_sources:")
foreach(i ${wheels_test_sources})
	message (STATUS ${i})  
endforeach()
find_package(GTest REQUIRED)
enable_testing ()
include_directories (${GTEST_INCLUDE_DIR})
wheels_add_executable(WheelsUnitTest ${wheels_test_sources} ./unittest.cpp)
target_link_libraries (WheelsUnitTest Wheels)
target_link_libraries (WheelsUnitTest optimized ${GTEST_LIBRARY})
target_link_libraries (WheelsUnitTest optimized ${GTEST_MAIN_LIBRARY})
target_link_libraries (WheelsUnitTest debug ${GTEST_LIBRARY_DEBUG})
target_link_libraries (WheelsUnitTest debug ${GTEST_MAIN_LIBRARY_DEBUG})