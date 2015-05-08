file(GLOB Common_HEADERS "*.h")


cuda_add_library(Common STATIC ${Common_HEADERS})
