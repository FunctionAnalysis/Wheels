PROJECT (Wheels)
cmake_minimum_required (VERSION 3.0)


set (CMAKE_ALLOW_LOOSE_CONSTRUCTS true)
list (APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)


if (${CLANG_VERSION_STRING})
    if(CLANG_VERSION_STRING VERSION_GREATER 3.4)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
    endif()
endif ()


set (DEPENDENCY_INCLUDES "")
set (DEPENDENCY_LIBS "")
set (DEPENDENCY_BIN_PATHS "")


find_package(Eigen3)
if (${EIGEN3_FOUND})
    list (APPEND DEPENDENCY_INCLUDES ${EIGEN3_INCLUDE_DIR})
    if (MSVC)
        add_definitions ( "/Dwheels_with_eigen" )
    else ()
        add_definitions ( "-Dwheels_with_eigen" )
    endif ()
endif()

find_package(Armadillo)
if (${ARMADILLO_FOUND})
    list (APPEND DEPENDENCY_INCLUDES ${ARMADILLO_INCLUDE_DIRS})
    list (APPEND DEPENDENCY_LIBS ${ARMADILLO_LIBRARIES})
    list (APPEND DEPENDENCY_BIN_PATHS ${ARMADILLO_LIBRARIES})
    if (MSVC)
        add_definitions ( "/Dwheels_with_armadillo" )
    else ()
        add_definitions ( "-Dwheels_with_armadillo" )
    endif ()
endif()

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


find_package(CUDA)
if (${CUDA_FOUND})  
    list (APPEND DEPENDENCY_INCLUDES ${CUDA_TOOLKIT_INCLUDE})
    if (MSVC)
        add_definitions ( "/Dwheels_with_cuda" )
    else ()
        add_definitions ( "-Dwheels_with_cuda" )
    endif ()
endif()


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

find_package(OpenCV REQUIRED)
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

find_package (Cereal REQUIRED)
if (${Cereal_FOUND})
    list (APPEND DEPENDENCY_INCLUDES ${CEREAL_INCLUDE_DIR}) 
    if (MSVC)
        add_definitions ( "/Dwheels_with_cereal" )
    else ()
        add_definitions ( "-Dwheels_with_cereal" )
    endif ()
endif()


if (${MSVC})
    add_definitions ("/Dwheels_with_stdfilesystem")
endif()



get_filename_component(wheels_data_dir "${CMAKE_CURRENT_SOURCE_DIR}/data/" REALPATH)
message (STATUS "data directory: " ${wheels_data_dir})
if (MSVC)
    add_definitions ( "/DNOMINMAX /W3 /D_CRT_SECURE_NO_WARNINGS /wd4503 /D_USE_MATH_DEFINES /D_SCL_SECURE_NO_WARNINGS")
    add_definitions ( "/Dwheels_data_dir_str=\"${wheels_data_dir}\"" )
else ()
    add_definitions ( "-Dwheels_data_dir_str=\"${wheels_data_dir}\"" )
endif ()


macro(wheels_add_executable BIN_NAME)
    if (CUDA_TOOLKIT_INCLUDE)
        cuda_add_executable(${BIN_NAME} ${ARGN})
    else()
        add_executable(${BIN_NAME} ${ARGN})
    endif()
endmacro()
macro(wheels_add_library BIN_NAME)
    if (CUDA_TOOLKIT_INCLUDE)
        cuda_add_library(${BIN_NAME} ${ARGN})
    else()
        add_library(${BIN_NAME} ${ARGN})
    endif()
endmacro()


include_directories (".")

add_subdirectory (wheels)