cmake_minimum_required(VERSION 3.0.0)
project(SOILProxy C CXX)
include(ExternalProject)

ExternalProject_Add(soil
     URL "file://${CMAKE_CURRENT_SOURCE_DIR}/Simple-OpenGL-Image-Library-master.zip"
     CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
     PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
     STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}"
# Disable install step
     INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(soil source_dir)
set(SOIL_INCLUDES ${source_dir}/src PARENT_SCOPE)

# Specify soil libs
ExternalProject_Get_Property(soil binary_dir)
set(SOIL_LIBS_DIR ${binary_dir})
if(NOT MSVC)
    set(SOIL_LIBS 
        ${SOIL_LIBS_DIR}/libSOIL.a
        PARENT_SCOPE)
else()
    set(SOIL_LIBS
        debug "${SOIL_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}SOIL${CMAKE_FIND_LIBRARY_SUFFIXES}"
        optimized "${SOIL_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}SOIL${CMAKE_FIND_LIBRARY_SUFFIXES}"
        PARENT_SCOPE)
endif()