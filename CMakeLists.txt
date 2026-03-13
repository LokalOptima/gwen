cmake_minimum_required(VERSION 3.24)
project(gwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA flags — hyper-optimized for RTX 5070 Ti (SM_120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --use_fast_math")

# CUTLASS (header-only)
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/cutlass/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cutlass/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include)
    message(STATUS "CUTLASS found at ${CMAKE_SOURCE_DIR}/third_party/cutlass")
else()
    message(WARNING "CUTLASS not found. GEMM kernels will use fallback. Run: git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass")
endif()

# Project includes
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Library: gwen core ---
file(GLOB_RECURSE GWEN_SOURCES
    src/kernels/*.cu
    src/gguf.cu
    src/model.cu
    src/memory.cu
    src/inference.cu
    src/tokenizer.cu
)

if(GWEN_SOURCES)
    add_library(gwen_core STATIC ${GWEN_SOURCES})
    target_include_directories(gwen_core PUBLIC include)
    target_link_libraries(gwen_core)
    set_target_properties(gwen_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- Main executable ---
if(EXISTS ${CMAKE_SOURCE_DIR}/src/main.cu)
    add_executable(gwen src/main.cu)
    target_link_libraries(gwen gwen_core)
    set_target_properties(gwen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- Server executable ---
if(EXISTS ${CMAKE_SOURCE_DIR}/src/server.cpp)
    find_package(CUDAToolkit REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(gwen_server src/server.cpp
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib/httplib.cpp)
    target_link_libraries(gwen_server gwen_core CUDA::cudart Threads::Threads)
    target_include_directories(gwen_server PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)
endif()

# --- Micro-benchmark ---
if(EXISTS ${CMAKE_SOURCE_DIR}/bench/micro_bench.cu)
    add_executable(micro_bench bench/micro_bench.cu)
    target_link_libraries(micro_bench gwen_core)
endif()

# --- Kernel tests ---
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_kernels.cu)
    add_executable(test_kernels tests/test_kernels.cu)
    target_link_libraries(test_kernels gwen_core)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_layers.cu)
    add_executable(test_layers tests/test_layers.cu)
    target_link_libraries(test_layers gwen_core)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_gemm.cu)
    add_executable(test_gemm tests/test_gemm.cu)
    target_link_libraries(test_gemm gwen_core)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_dp4a.cu)
    add_executable(test_dp4a tests/test_dp4a.cu)
    target_link_libraries(test_dp4a gwen_core)
endif()

# --- Profiler ---
if(EXISTS ${CMAKE_SOURCE_DIR}/bench/profile_forward.cu)
    add_executable(profile_forward bench/profile_forward.cu)
    target_link_libraries(profile_forward gwen_core)
    set_target_properties(profile_forward PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
