set(hints
  $ENV{EXTERNLIBS}/glut
  $ENV{EXTERNLIBS}/freeglut
  $ENV{LIB_BASE_PATH}/
  $ENV{LIB_BASE_PATH}/glut/
  $ENV{LIB_BASE_PATH}/freeglut/
)

set(paths
  /usr
  /usr/local
)

find_path(GLUT_INCLUDE_DIR
  NAMES
    GL/glut.h
    GL/freeglut.h
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
)

find_library(GLUT_LIBRARY
  NAMES
    GLUT # MacOS X Framework
    glut
    glut_static
    gluts
    glut32
    glut32_static
    glut32s
    freeglut
    freeglut_static
    freegluts
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
	lib/x64
    lib
)

set(GLUT_LIBRARIES ${GLUT_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLUT DEFAULT_MSG
  GLUT_INCLUDE_DIR
  GLUT_LIBRARY
)

if (NOT GLUT_FOUND)
    message(FATAL_ERROR "GLUT required, but not found!  ${GLUT_LIBRARIES}")
endif()

mark_as_advanced(GLUT_INCLUDE_DIR GLUT_LIBRARY)
