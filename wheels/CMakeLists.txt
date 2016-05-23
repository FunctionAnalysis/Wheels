set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (wheels_sources "")
set (wheels_includes "")
set (wheels_test_sources "")

# add essential modules
set (modules core tensor)
if (${Cereal_FOUND})
    list (APPEND modules serialize)
endif ()
if (${OpenBLAS_FOUND})
    list (APPEND modules math)
endif()

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

if (MSVC)
    message (WARNING "CMake cannot configure Visual Studio to"
        " modify the environment path during program execution, "
        "therefore you have to do this manually: "
        "add 'PATH=\$(PATH);${DEPENDENCY_BIN_PATHS};' "
        "to [Project Property]->[Debug]->[Environment] ")
endif ()


# the lib project
message(STATUS "wheels_sources:")
foreach(i ${wheels_sources})
    message (STATUS ${i})  
endforeach()
add_library(Wheels.Lib ${wheels_includes} ${wheels_sources} ./dummy.cpp)
target_link_libraries (Wheels.Lib ${DEPENDENCY_LIBS})


# the test project
if (${BuildWheelsUnitTest})

    enable_testing()
    include_directories(${GTEST_INCLUDES})
    
    add_executable(Wheels.UnitTest ${wheels_test_sources} ./unittest.cpp)
    target_link_libraries(Wheels.UnitTest Wheels.Lib ${DEPENDENCY_LIBS})
    target_link_libraries(Wheels.UnitTest ${GTEST_LIBS})
    add_dependencies(Wheels.UnitTest googletest)

endif ()