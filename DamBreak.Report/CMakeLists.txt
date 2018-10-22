file(GLOB DamBreakReport_SRCS    "*.cpp")
file(GLOB DamBreakReport_HEADERS "*.h")
add_executable(DamBreakReport ${DamBreakReport_SRCS} ${DamBreakReport_HEADERS})
target_link_libraries(DamBreakReport DamBreakCore ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
