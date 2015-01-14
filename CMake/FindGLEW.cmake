set(hints
  $ENV{EXTERNLIBS}/glew  
  $ENV{LIB_BASE_PATH}/  
  $ENV{LIB_BASE_PATH}/glew/
)

set(paths
  /usr
  /usr/local
)

find_path(GLEW_INCLUDE_DIR
  NAMES
    GL/glew.h    
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    include
)

find_library(GLEW_LIBRARY
  NAMES
    GLEW glew32 glew glew32s
  HINTS
    ${hints}
  PATHS
    ${paths}
  PATH_SUFFIXES
    lib64
    lib
    lib/Release
    lib/Release/x64
    lib/Release/Win32
)

set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIR})
set(GLEW_LIBRARIES ${GLEW_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW
    REQUIRED_VARS GLEW_INCLUDE_DIR GLEW_LIBRARY)

if(GLEW_FOUND AND NOT TARGET GLEW::GLEW)
  add_library(GLEW::GLEW UNKNOWN IMPORTED)
  set_target_properties(GLEW::GLEW PROPERTIES
    IMPORTED_LOCATION "${GLEW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")
endif()

if (NOT GLEW_FOUND)	
    message(FATAL_ERROR "GLEW required, but not found!  ${GLEW_LIBRARIES}")
endif()

mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY)
