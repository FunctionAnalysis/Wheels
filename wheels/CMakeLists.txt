set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (wheels_sources "")
set (wheels_test_sources "")

file (GLOB wheels_sources 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)
file (GLOB wheels_test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.test.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.test.hpp"
)
list (REMOVE_ITEM wheels_sources ${wheels_test_sources})
source_group ("src" FILES ${wheels_sources})
source_group ("src" FILES ${wheels_test_sources})

# add unsupported modules
set (unsupported_modules "eigen")
if (${MATLAB_FOUND})
    list (APPEND unsupported_modules matlab)
endif ()
if (${OpenCV_FOUND})
    list (APPEND unsupported_modules opencv)
endif ()
if (${OpenBLAS_FOUND})
    list (APPEND unsupported_modules auxmath)
endif()

foreach (M ${unsupported_modules})
    file (GLOB "Src"
        "${CMAKE_CURRENT_SOURCE_DIR}/unsupported/${M}/*.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/unsupported/${M}/*.hpp"
    )
    file (GLOB "TestSrc" 
        "${CMAKE_CURRENT_SOURCE_DIR}/unsupported/${M}/*.test.cpp" 
        "${CMAKE_CURRENT_SOURCE_DIR}/unsupported/${M}/*.test.hpp"
    )
    list (REMOVE_ITEM Src ${TestSrc})
    source_group ("unsupported\\${M}" FILES ${Src})
    source_group ("unsupported\\${M}" FILES ${TestSrc})
    list (APPEND wheels_sources ${Src})
    list (APPEND wheels_test_sources ${TestSrc})
endforeach()

if (MSVC)
    message (WARNING "CMake cannot configure Visual Studio to"
        " modify the environment path during program execution, "
        "therefore you have to do this manually: "
        "add 'PATH=\$(PATH);${DEPENDENCY_BIN_PATHS};' "
        "to [Project Property]->[Debug]->[Environment] ")
endif ()

foreach (i ${DEPENDENCY_INCLUDES})
    message (STATUS "DEPENDENCY_INCLUDES: ${i}")
endforeach ()


# the lib project
message(STATUS "wheels_sources:")
foreach(i ${wheels_sources})
    message (STATUS ${i})  
endforeach()
add_library(Wheels.Lib ${wheels_includes} ${wheels_sources} ./dummy.cpp)
target_include_directories (Wheels.Lib PUBLIC ${DEPENDENCY_INCLUDES})
target_link_libraries (Wheels.Lib ${DEPENDENCY_LIBS})
add_dependencies(Wheels.Lib ${DEPENDENCY_NAMES})


# the test project
if (${BuildUnitTest})
    enable_testing()    
    add_executable(Wheels.UnitTest ${wheels_test_sources} ./unittest.cpp)
    target_include_directories (Wheels.UnitTest PUBLIC
        ${DEPENDENCY_INCLUDES} ${TEST_DEPENDENCY_INCLUDES})
    target_link_libraries (Wheels.UnitTest Wheels.Lib)
    target_link_libraries (Wheels.UnitTest ${DEPENDENCY_LIBS})
    target_link_libraries (Wheels.UnitTest ${TEST_DEPENDENCY_LIBS})
    add_dependencies(Wheels.UnitTest Wheels.Lib 
        ${DEPENDENCY_NAMES} ${TEST_DEPENDENCY_NAMES})
endif ()