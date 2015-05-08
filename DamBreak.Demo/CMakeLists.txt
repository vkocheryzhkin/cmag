file(GLOB DamBreakDemo_SRCS    "*.cpp")
file(GLOB DamBreakDemo_HEADERS "*.h")

add_executable(DamBreakDemo ${DamBreakDemo_SRCS} ${DamBreakDemo_HEADERS})

target_link_libraries(DamBreakDemo DamBreakCore ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
