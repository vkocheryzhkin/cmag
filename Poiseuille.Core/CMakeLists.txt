if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

file(GLOB PoiseuilleCore_SRCS    "*.cpp" "*.cu")
file(GLOB PoiseuilleCore_HEADERS "*.h" "*.cuh")

cuda_add_library(PoiseuilleCore STATIC ${PoiseuilleCore_SRCS} ${PoiseuilleCore_HEADERS})
#set_target_properties(PoiseuilleCore PROPERTIES DEBUG_POSTFIX "d")
