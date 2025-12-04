cmake_minimum_required(VERSION 3.5)
project(minimal_test_working LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

add_executable(repro_test minimal_test.hip)
target_include_directories(repro_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../example/ck_tile/01_fmha
)

target_compile_options(repro_test PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O0>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>
    $<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>
    -fPIE
)

set_target_properties(repro_test PROPERTIES LINKER_LANGUAGE CXX)

