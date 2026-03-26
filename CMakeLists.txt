cmake_minimum_required(VERSION 3.24)
project(gwen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA flags — hyper-optimized for RTX 5070 Ti (SM_120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --use_fast_math --extended-lambda")

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
file(GLOB GWEN_KERNEL_SOURCES src/kernels/*.cu)
list(FILTER GWEN_KERNEL_SOURCES EXCLUDE REGEX "fattn_mma\\.cu$")
list(FILTER GWEN_KERNEL_SOURCES EXCLUDE REGEX "gemm_mmq\\.cu$")   # compiled separately with vendored headers

set(GWEN_SOURCES
    ${GWEN_KERNEL_SOURCES}
    src/gguf.cu
    src/model.cu
    src/inference.cu
    src/tokenizer.cu
)

if(GWEN_SOURCES)
    find_package(CUDAToolkit REQUIRED)
    add_library(gwen_core STATIC ${GWEN_SOURCES})
    target_include_directories(gwen_core PUBLIC include)
    target_link_libraries(gwen_core CUDA::cublas)
    set_target_properties(gwen_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- MMA flash attention (vendored from llama.cpp) ---
if(EXISTS ${CMAKE_SOURCE_DIR}/src/kernels/fattn_mma.cu)
    add_library(gwen_fattn STATIC src/kernels/fattn_mma.cu)
    target_include_directories(gwen_fattn PRIVATE include)
    set_target_properties(gwen_fattn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(gwen_core gwen_fattn)
endif()

# --- MMQ fused K-quant GEMM (vendored from llama.cpp) ---
if(EXISTS ${CMAKE_SOURCE_DIR}/src/kernels/gemm_mmq.cu)
    add_library(gwen_mmq STATIC src/kernels/gemm_mmq.cu)
    target_include_directories(gwen_mmq PRIVATE include)
    set_target_properties(gwen_mmq PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(gwen_core gwen_mmq)
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
        ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib/httplib.cpp)
    target_link_libraries(gwen_server gwen_core CUDA::cudart Threads::Threads)
    target_include_directories(gwen_server PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib)
endif()

# --- Kernel tests ---
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_kernels.cu)
    add_executable(test_kernels tests/test_kernels.cu)
    target_link_libraries(test_kernels gwen_core)
endif()

# --- Benchmark (llama-bench replica) ---
if(EXISTS ${CMAKE_SOURCE_DIR}/bench/gwen_bench.cu)
    add_executable(gwen_bench bench/gwen_bench.cu)
    target_link_libraries(gwen_bench gwen_core)
    set_target_properties(gwen_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# --- Profiler ---
if(EXISTS ${CMAKE_SOURCE_DIR}/bench/profile_forward.cu)
    add_executable(profile_forward bench/profile_forward.cu)
    target_link_libraries(profile_forward gwen_core)
    set_target_properties(profile_forward PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
