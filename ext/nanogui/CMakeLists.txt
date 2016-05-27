cmake_minimum_required(VERSION 3.0.0)
project(NanoGUIProxy C CXX)
include(ExternalProject)

ExternalProject_Add(nanogui
    URL "file://${CMAKE_CURRENT_SOURCE_DIR}/nanogui.zip"
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_COMMAND ""
)

# Specify includes
ExternalProject_Get_Property(nanogui source_dir)
set(NANOGUI_INCLUDES ${source_dir}/include 
    ${source_dir}/ext/eigen
    ${source_dir}/ext/glad/include
    ${source_dir}/ext/glfw/include
    ${source_dir}/ext/nanovg/src
    PARENT_SCOPE)

# Specify libs
ExternalProject_Get_Property(nanogui binary_dir)
set(NANOGUI_LIBS_DIR ${binary_dir})
if(NOT MSVC)
    set(NANOGUI_LIBS 
        "${NANOGUI_LIBS_DIR}/libnanogui.a" 
        PARENT_SCOPE)
else()
    set(NANOGUI_LIBS
        debug "${NANOGUI_LIBS_DIR}/Debug/${CMAKE_FIND_LIBRARY_PREFIXES}nanogui${CMAKE_FIND_LIBRARY_SUFFIXES}"
        optimized "${NANOGUI_LIBS_DIR}/Release/${CMAKE_FIND_LIBRARY_PREFIXES}nanogui${CMAKE_FIND_LIBRARY_SUFFIXES}"
        PARENT_SCOPE)
endif()


