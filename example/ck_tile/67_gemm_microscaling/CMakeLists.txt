set(TARGET_NAME tile_example_gemm_mx)

set(TARGET_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND TARGET_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
# list(APPEND TARGET_COMPILE_OPTIONS --save-temps)

add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL gemm_mx.cpp)
target_compile_options(${TARGET_NAME} PRIVATE ${TARGET_COMPILE_OPTIONS})
