if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

file(GLOB PoiseuilleDemo_SRCS    "*.cpp")
file(GLOB PoiseuilleDemo_HEADERS "*.h")

add_executable(PoiseuilleDemo ${PoiseuilleDemo_SRCS} ${PoiseuilleDemo_HEADERS})
target_link_libraries(PoiseuilleDemo PoiseuilleCore ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES})
