cmake_minimum_required(VERSION 3.24)

project(
    BlackForge
    VERSION 0.1.0
    DESCRIPTION "Vertical ML language and compiler for NVIDIA GPUs"
    LANGUAGES CXX

)

# config globale

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# usa relase quando non viene specificata una config

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

# opzioni progetto

option(
    BLACKFORGE_ENABLE_CUDA
    "Enable the NVIDIA CUDA backend"
    ON
)

option(
    BLACKFORGE_BUILD_TESTS
    "Build BlackForge tests"
    OFF
)

option(
    BLACKFORGE_ENABLE_WARNINGS
    "Enable strick compiler warnings"
    ON
)

# per le mie amate blackwell

set(
    BLACKFORGE_CUDA_ARCHITECTURES
    "120"
    CACHE STRING
    "CUDA architectures targeted by BlackForge"
)

# detect cuda

if(BLACKFORGE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        message(
            STATUS
            "Blackforge CUDA backend enabled"
        )

        message(
            STATUS
            "CUDA compiler: ${CMAKE_CUDA_COMPILER}"
        )

        message(
            STATUS
            "CUDA architectures: ${BLACKFORGE_CUDA_ARCHITECTURES}"
        )

        find_package(CUDAToolkit REQUIRED)
    else()
        message(
            WARNING
            "CUDA compiler not found. Building BlackForge without CUDA."
        )

        set(BLACKFORGE_ENABLE_CUDA OFF)
    endif()
endif()

# core library
#
# L'elenco cresce milestone dopo milestone: qui compaiono solo i file
# realmente implementati, non placeholder futuri.

set(
    BLACKFORGE_CORE_SOURCES
    src/diagnostics/diagnostic.cpp
    src/frontend/token.cpp
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/ast/ast.cpp
    src/sema/dtype.cpp
    src/sema/semantic_analyzer.cpp
    src/ir/value.cpp
    src/ir/operation.cpp
    src/ir/module.cpp
    src/ir/ir_builder.cpp
    src/runtime/tensor.cpp
    src/runtime/device.cpp
    src/backend/cpu/ops.cpp
    src/backend/cpu/executor.cpp
    src/backend/cpu/random_init.cpp
    src/backend/cpu/autodiff.cpp
    src/backend/cpu/loss.cpp
    src/backend/cpu/model.cpp
    src/backend/cpu/optimizer.cpp
    src/backend/cpu/checkpoint.cpp
    src/backend/cpu/train_runner.cpp
    src/backend/cpu/forecast_runner.cpp
    src/backend/cpu/benchmark.cpp
    src/backend/cpu/quantize.cpp
    src/data/dataset.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/dataset_prep.cpp
)

if(BLACKFORGE_ENABLE_CUDA)
    list(
        APPEND BLACKFORGE_CORE_SOURCES
        src/backend/cuda/device_pool.cu
        src/backend/cuda/device_tensor.cu
        src/backend/cuda/ops_elementwise.cu
        src/backend/cuda/ops_gemm.cu
        src/backend/cuda/ops_tensorcore.cu
        src/backend/cuda/attention_batched.cu
        src/backend/cuda/ops_transformer.cu
        src/backend/cuda/executor.cu
        src/backend/cuda/device_query.cu
        src/backend/cuda/autodiff.cu
        src/backend/cuda/loss.cu
        src/backend/cuda/model.cu
        src/backend/cuda/checkpoint.cu
        src/backend/cuda/optimizer.cu
        src/backend/cuda/train_runner.cu
        src/backend/cuda/multi_gpu_train_runner.cu
    )
endif()

add_library(
    blackforge_core
    STATIC
    ${BLACKFORGE_CORE_SOURCES}
)

target_include_directories(
    blackforge_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_definitions(
    blackforge_core
    PUBLIC
        BLACKFORGE_VERSION="${PROJECT_VERSION}"
)

if(BLACKFORGE_ENABLE_CUDA)
    set_target_properties(
        blackforge_core
        PROPERTIES
            CUDA_ARCHITECTURES
                "${BLACKFORGE_CUDA_ARCHITECTURES}"

            CUDA_SEPARABLE_COMPILATION
                ON
    )

    target_compile_definitions(
        blackforge_core
        PUBLIC
            BLACKFORGE_HAS_CUDA=1
    )

    target_link_libraries(
        blackforge_core
        PUBLIC
            CUDA::cudart
            CUDA::cublas
            CUDA::cublasLt
    )
else()
    target_compile_definitions(
        blackforge_core
        PUBLIC
            BLACKFORGE_HAS_CUDA=0
    )
endif()

# warnings

if(BLACKFORGE_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(
            blackforge_core
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    /W4
                    /permissive-
                    /EHsc
                >
        )
    else()
        target_compile_options(
            blackforge_core
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -Wall
                    -Wextra
                    -Wpedantic
                >
        )
    endif()
endif()

# ottimizzazioni

if(MSVC)
    target_compile_options(
        blackforge_core
        PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    )
else()
    target_compile_options(
        blackforge_core
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
endif()

if(BLACKFORGE_ENABLE_CUDA)
    target_compile_options(
        blackforge_core
        PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math --extra-device-vectorization>
    )
endif()

# main

add_executable(
    blackforge
    src/main.cpp
)

target_link_libraries(
    blackforge
    PRIVATE
        blackforge_core
)

# tests

if(BLACKFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "")
message(STATUS "Blackforge ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++23")
message(STATUS "CUDA backend: ${BLACKFORGE_ENABLE_CUDA}")
message(STATUS "Build tests: ${BLACKFORGE_BUILD_TESTS}")
message(STATUS "")
