project(cmag)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

cmake_minimum_required(VERSION 2.6)
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" )
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenGL
find_package(OpenGL REQUIRED)

# GLUT
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

ADD_SUBDIRECTORY( Common )

ADD_SUBDIRECTORY( Poiseuille.Core )
ADD_SUBDIRECTORY( Poiseuille.Demo )
ADD_SUBDIRECTORY( Poiseuille.Report )


#ADD_SUBDIRECTORY( DamBreak.Core )
#ADD_SUBDIRECTORY( DamBreak.Demo )
