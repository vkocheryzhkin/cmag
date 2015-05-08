file(GLOB PoiseuilleReport_SRCS    "*.cpp")
file(GLOB PoiseuilleReport_HEADERS "*.h")
add_executable(PoiseuilleReport ${PoiseuilleReport_SRCS} ${PoiseuilleReport_HEADERS})
target_link_libraries(PoiseuilleReport PoiseuilleCore ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
