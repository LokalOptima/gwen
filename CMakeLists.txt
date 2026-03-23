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

# --- llama.cpp source path (for MMA flash attention kernel) ---
set(LLAMA_CPP_PATH "$ENV{HOME}/git/llama.cpp" CACHE PATH "Path to llama.cpp source tree")

# --- Library: gwen core ---
file(GLOB GWEN_KERNEL_SOURCES src/kernels/*.cu)
list(FILTER GWEN_KERNEL_SOURCES EXCLUDE REGEX "fattn_mma\\.cu$")
list(FILTER GWEN_KERNEL_SOURCES EXCLUDE REGEX "gemm_mmq\\.cu$")   # legacy Q4_K GEMM, no longer linked
list(FILTER GWEN_KERNEL_SOURCES EXCLUDE REGEX "gemm_fp8\\.cu$")   # separate lib with 120a arch

set(GWEN_SOURCES
    ${GWEN_KERNEL_SOURCES}
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

# --- MMA flash attention (compiled against llama.cpp headers) ---
if(EXISTS ${LLAMA_CPP_PATH}/ggml/src/ggml-cuda/common.cuh)
    add_library(gwen_fattn STATIC src/kernels/fattn_mma.cu)
    target_include_directories(gwen_fattn PRIVATE
        include
        ${LLAMA_CPP_PATH}/ggml/include
        ${LLAMA_CPP_PATH}/ggml/src
        ${LLAMA_CPP_PATH}/ggml/src/ggml-cuda
    )
    set_target_properties(gwen_fattn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(gwen_core gwen_fattn)
    message(STATUS "llama.cpp found — MMA flash attention enabled")

    # --- FP8 GEMM (CUTLASS 3.x SM120 blockwise scaling) ---
    # Requires 120a for block-scaled MMA instructions (SM120 "all features")
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/kernels/gemm_fp8.cu)
        add_library(gwen_fp8_gemm STATIC src/kernels/gemm_fp8.cu)
        target_include_directories(gwen_fp8_gemm PRIVATE include)
        set_target_properties(gwen_fp8_gemm PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "120a")
        target_link_libraries(gwen_core gwen_fp8_gemm)
        message(STATUS "FP8 GEMM (CUTLASS 3.x SM120a) enabled")
    endif()
else()
    message(WARNING "llama.cpp not found at ${LLAMA_CPP_PATH} — MMA flash attention disabled")
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

# --- Dev Server executable (MTP v3 training) ---
if(EXISTS ${CMAKE_SOURCE_DIR}/src/dev_server.cpp)
    find_package(CUDAToolkit REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(gwen_dev_server src/dev_server.cpp
        ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib/httplib.cpp)
    target_link_libraries(gwen_dev_server gwen_core CUDA::cudart Threads::Threads)
    target_include_directories(gwen_dev_server PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib)
endif()

# --- Micro-benchmarks ---
if(EXISTS ${CMAKE_SOURCE_DIR}/bench/micro_bench.cu)
    add_executable(micro_bench bench/micro_bench.cu)
    target_link_libraries(micro_bench gwen_core)
endif()

# bench_gemm_approaches: disabled (references removed gwen_gemm_mmq Q4_K path)

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

