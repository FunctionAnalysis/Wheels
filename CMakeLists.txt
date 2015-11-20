PROJECT (Wheels)
cmake_minimum_required (VERSION 3.0)


option (WITH_EIGEN "with EIGEN" off)
option (WITH_CUDA "with CUDA" off)
option (WITH_MATLAB "with MATLAB" off)
option (WITH_OPENCV "with OPENCV" off)


set (CMAKE_ALLOW_LOOSE_CONSTRUCTS true)
list (APPEND CMAKE_MODULE_PATH 
	${CMAKE_CURRENT_SOURCE_DIR}/cmake
)


if (NOT DEFINED MSVC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif ()


set (DEPENDENCY_INCLUDES "")
set (DEPENDENCY_LIBS "")
set (DEPENDENCY_BIN_PATHS "")

if (${WITH_EIGEN})
	find_package(Eigen3 REQUIRED)
	list (APPEND DEPENDENCY_INCLUDES ${EIGEN3_INCLUDE_DIR})
	if (MSVC)
		add_definitions ( "/Dwheels_with_eigen" )
	else ()
		add_definitions ( "-Dwheels_with_eigen" )
	endif ()
endif()

if (${WITH_CUDA})
	find_package(CUDA REQUIRED)
	list (APPEND DEPENDENCY_INCLUDES ${CUDA_TOOLKIT_INCLUDE})
	if (MSVC)
		add_definitions ( "/Dwheels_with_cuda" )
	else ()
		add_definitions ( "-Dwheels_with_cuda" )
	endif ()
endif()

if (${WITH_MATLAB})
	find_package(MATLAB REQUIRED)
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

if (${WITH_OPENCV})
	find_package(OpenCV REQUIRED)
	list (APPEND DEPENDENCY_INCLUDES ${OpenCV_INCLUDE_DIRS})
	list (APPEND DEPENDENCY_LIBS ${OpenCV_LIBS})
	list (APPEND DEPENDENCY_BIN_PATHS ${_OpenCV_LIB_PATH})
endif()


get_filename_component(wheels_data_dir "${CMAKE_CURRENT_SOURCE_DIR}/data/" REALPATH)
message (STATUS "data directory: " ${wheels_data_dir})
if (MSVC)
	add_definitions ( "/DNOMINMAX")
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