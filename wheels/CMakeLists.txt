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

file (GLOB 3rd_sources 
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd/*.cu" 
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd/*.cuh")
source_group ("3rd" FILES ${3rd_sources})



set (wheels_sources 
    ${core_sources} 
    ${tensor_sources} 
    ${3rd_sources})
file (GLOB_RECURSE wheels_test_sources "." *.test.cpp *.test.hpp *.test.cu *.test.cuh)
# remove test sources from the lib sources
list (REMOVE_ITEM wheels_sources ${wheels_test_sources})


include_directories (${DEPENDENCY_INCLUDES})


# the lib project
message(STATUS "wheels_sources:")
foreach(i ${wheels_sources})
    message (STATUS ${i})  
endforeach()
wheels_add_library(Wheels.Lib ${wheels_sources} ./dummy.cpp)
target_link_libraries (Wheels.Lib ${DEPENDENCY_LIBS})



if (MSVC)
    message (WARNING "CMake cannot configure Visual Studio to modify the environment path during program execution, "
        "therefore you have to do this manually: add 'PATH=\$(PATH);${DEPENDENCY_BIN_PATHS};' to [Project Property]->[Debug]->[Environment] ")
endif ()


# the test project
message(STATUS "wheels_test_sources:")
foreach(i ${wheels_test_sources})
    message (STATUS ${i})  
endforeach()
find_package(GTest REQUIRED)
enable_testing ()

include_directories (${GTEST_INCLUDE_DIR})
wheels_add_executable(Wheels.UnitTest ${wheels_test_sources} ./unittest.cpp)
target_link_libraries (Wheels.UnitTest ${DEPENDENCY_LIBS})
target_link_libraries (Wheels.UnitTest Wheels.Lib)
target_link_libraries (Wheels.UnitTest optimized ${GTEST_LIBRARY})
target_link_libraries (Wheels.UnitTest optimized ${GTEST_MAIN_LIBRARY})
target_link_libraries (Wheels.UnitTest debug ${GTEST_LIBRARY_DEBUG})
target_link_libraries (Wheels.UnitTest debug ${GTEST_MAIN_LIBRARY_DEBUG})