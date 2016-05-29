cmake_minimum_required (VERSION 3.0)
PROJECT (Wheels VERSION 0.1 LANGUAGES CXX)

option (USE_OPENBLAS "use OpenBLAS" off)
option (USE_MATLAB "use MATLAB" off)
option (USE_OPENCV "use OpenCV" off)

option (BuildUnitTest "build UnitTest" on)

set (CMAKE_ALLOW_LOOSE_CONSTRUCTS true)
list (APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    message (STATUS "Clang: ${CLANG_VERSION_STRING}")
    if(CLANG_VERSION_STRING VERSION_GREATER 3.5)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
    endif()
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    message (STATUS "GCC: ${GCC_VERSION}")
    if (NOT (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9))
        message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.9 or greater.")
    endif ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

add_subdirectory(ext)

if (${USE_OPENBLAS})
find_package(OpenBLAS)
if (${OpenBLAS_FOUND})
	list (APPEND DEPENDENCY_INCLUDES ${OpenBLAS_INCLUDE_DIR})
    list (APPEND DEPENDENCY_LIBS ${OpenBLAS_LIB})
    list (APPEND DEPENDENCY_BIN_PATHS ${OpenBLAS_DIR}/bin)
    if (MSVC)
        add_definitions ( "/Dwheels_with_openblas" )
    else ()
        add_definitions ( "-Dwheels_with_openblas" )
    endif ()
endif()
endif()

if (${USE_MATLAB})
find_package(MATLAB)
if (${MATLAB_FOUND})
    #  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
    #  MATLAB_LIBRARIES:   required libraries: libmex, etc
    #  MATLAB_MEX_LIBRARY: path to libmex.lib
    #  MATLAB_MX_LIBRARY:  path to libmx.lib
    #  MATLAB_MAT_LIBRARY:  path to libmat.lib # added
    #  MATLAB_ENG_LIBRARY: path to libeng.lib
    #  MATLAB_ROOT: path to Matlab's root directory
    list (APPEND DEPENDENCY_INCLUDES ${MATLAB_INCLUDE_DIR})
    list (APPEND DEPENDENCY_LIBS ${MATLAB_LIBRARIES})
    list (APPEND DEPENDENCY_LIBS ${MATLAB_MAT_LIBRARY})
    if (MSVC)
        add_definitions ( "/Dwheels_with_matlab" )
    else ()
        add_definitions ( "-Dwheels_with_matlab" )
    endif ()
endif()
endif()

if (${USE_OPENCV})
find_package(OpenCV)
if (${OpenCV_FOUND})
    list (APPEND DEPENDENCY_INCLUDES ${OpenCV_INCLUDE_DIRS})
    list (APPEND DEPENDENCY_LIBS ${OpenCV_LIBS})
    list (APPEND DEPENDENCY_BIN_PATHS ${_OpenCV_LIB_PATH})
    if (MSVC)
        add_definitions ( "/Dwheels_with_opencv" )
    else ()
        add_definitions ( "-Dwheels_with_opencv" )
    endif ()
endif()
endif()

get_filename_component(wheels_data_dir "${CMAKE_CURRENT_SOURCE_DIR}/data/" REALPATH)
message (STATUS "data directory: " ${wheels_data_dir})
if (MSVC)
    add_definitions ( "/DNOMINMAX /W3 /wd4503 /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS")
    add_definitions ( "/Dwheels_data_dir_str=\"${wheels_data_dir}\"" )
else ()
	add_definitions ( "-D_USE_MATH_DEFINES" )
    add_definitions ( "-Dwheels_data_dir_str=\"${wheels_data_dir}\"" )
endif ()

foreach (i ${TEST_DEPENDENCY_LIBS})
    message(STATUS "TEST_DEPENDENCY_LIBS: ${i}")
endforeach()

add_subdirectory (wheels)
