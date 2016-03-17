set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (wheels_sources "")
set (wheels_includes "")
set (wheels_test_sources "")

# add essential modules
set (modules core tensor math)
if (${Cereal_FOUND})
    list (APPEND modules serialize)
endif ()

foreach (M ${modules})
    file (GLOB "Src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cu" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cuh"
    )
    file (GLOB "TestSrc" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.test.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.test.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.test.cu" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.test.cuh"
    )
    list (REMOVE_ITEM Src ${TestSrc})
    source_group ("modules\\${M}" FILES ${Src})
    source_group ("modules\\${M}" FILES ${TestSrc})
    list (APPEND wheels_sources ${Src})
    list (APPEND wheels_test_sources ${TestSrc})
    list (APPEND wheels_includes "${CMAKE_CURRENT_SOURCE_DIR}/${M}")
endforeach()
source_group ("includes" FILES ${wheels_includes})

# add unsupported modules
set (unsupported_modules "")
if (${MATLAB_FOUND})
    list (APPEND unsupported_modules matlab)
endif ()
if (${EIGEN3_FOUND})
    list (APPEND unsupported_modules eigen)
endif ()
if (${OpenCV_FOUND})
    list (APPEND unsupported_modules opencv)
endif ()

foreach (M ${unsupported_modules})
    file (GLOB "Src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.cu" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.cuh"
    )
    file (GLOB "TestSrc" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.test.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.test.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.test.cu" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/unsupported/${M}/*.test.cuh"
    )
    list (REMOVE_ITEM Src ${TestSrc})
    source_group ("modules\\unsupported\\${M}" FILES ${Src})
    source_group ("modules\\unsupported\\${M}" FILES ${TestSrc})
    list (APPEND wheels_sources ${Src})
    list (APPEND wheels_test_sources ${TestSrc})
endforeach()


include_directories (${DEPENDENCY_INCLUDES})

# the lib project
message(STATUS "wheels_sources:")
foreach(i ${wheels_sources})
    message (STATUS ${i})  
endforeach()
add_library(Wheels.Lib ${wheels_includes} ${wheels_sources} ./dummy.cpp)
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
add_executable(Wheels.UnitTest ${wheels_test_sources} ./unittest.cpp)
target_link_libraries (Wheels.UnitTest ${DEPENDENCY_LIBS})
target_link_libraries (Wheels.UnitTest Wheels.Lib)


if (EXISTS ${GTEST_LIBRARY_DEBUG})
    message (STATUS ${GTEST_LIBRARY_DEBUG})
    target_link_libraries (Wheels.UnitTest optimized ${GTEST_LIBRARY})
    target_link_libraries (Wheels.UnitTest debug ${GTEST_LIBRARY_DEBUG})
else ()
    target_link_libraries (Wheels.UnitTest ${GTEST_LIBRARY})
endif ()

if (EXISTS ${GTEST_MAIN_LIBRARY_DEBUG})
    message (STATUS ${GTEST_MAIN_LIBRARY_DEBUG})
    target_link_libraries (Wheels.UnitTest optimized ${GTEST_MAIN_LIBRARY})
    target_link_libraries (Wheels.UnitTest debug ${GTEST_MAIN_LIBRARY_DEBUG})
else ()
    target_link_libraries (Wheels.UnitTest ${GTEST_MAIN_LIBRARY})
endif ()