set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# all files
set (modules core tensor math unsupported)

if (${Cereal_FOUND})
    list (APPEND modules serialize)
endif ()

set (wheels_sources "")
foreach (M ${modules})
    file (GLOB "Src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cu" 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/${M}/*.cuh"
    )
    source_group ("modules\\${M}" FILES ${Src})
    list (APPEND wheels_sources ${Src})
endforeach()
set (wheels_includes "")
foreach (M ${modules})
    list (APPEND wheels_includes "${CMAKE_CURRENT_SOURCE_DIR}/${M}")
endforeach()
source_group ("includes" FILES ${wheels_includes})


file (GLOB_RECURSE wheels_test_sources "." *.test.cpp *.test.hpp *.test.cu *.test.cuh)
# remove test sources from the lib sources
list (REMOVE_ITEM wheels_sources ${wheels_test_sources})


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