file(GLOB DamBreakCore_SRCS    "*.cpp" "*.cu")
file(GLOB DamBreakCore_HEADERS "*.h" "*.cuh")

cuda_add_library(DamBreakCore STATIC ${DamBreakCore_SRCS} ${DamBreakCore_HEADERS})
