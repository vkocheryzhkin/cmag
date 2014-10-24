project(cmag)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

cmake_minimum_required(VERSION 2.6)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenGL
find_package(OpenGL REQUIRED)

# GLUT
set (CMAKE_PREFIX_PATH "C:\\Tools\\freeglut" )
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

ADD_SUBDIRECTORY( Poiseuille.Core )
