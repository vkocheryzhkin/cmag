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
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIR})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
    message(ERROR ${GLUT_ROOT_PATH})    
endif(NOT GLUT_FOUND)

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

ADD_SUBDIRECTORY( Poiseuille.Core )
ADD_SUBDIRECTORY( Poiseuille.Demo )
