cmake_minimum_required(VERSION 3.8)
project(llama_cpp_vendor)

include(FetchContent)
find_package(ament_cmake REQUIRED)

FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b5600
  GIT_SHALLOW    TRUE
)

option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)

FetchContent_MakeAvailable(llama)

# ggml
set_target_properties(
  ggml PROPERTIES
  OUTPUT_NAME "llama_ggml"
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>"
  CXX_STANDARD 17
)

set_target_properties(
  ggml-base PROPERTIES
  OUTPUT_NAME "llama_ggml_base"
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>"
  CXX_STANDARD 17
)

set_target_properties(
  ggml-cpu PROPERTIES
  OUTPUT_NAME "llama_ggml_cpu"
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>"
  CXX_STANDARD 17
)

if(GGML_CUDA)
  set_target_properties(
    ggml-cuda PROPERTIES
    OUTPUT_NAME "llama_ggml_cuda"
    INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>"
    CXX_STANDARD 17
  )
endif()

# llama
set_target_properties(
  build_info llama common PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/include;${llama_SOURCE_DIR}/src;${llama_SOURCE_DIR}/common;${llama_SOURCE_DIR}/>"
  CXX_STANDARD 17
)

# mtmd
add_library(mtmd
  ${llama_SOURCE_DIR}/tools/mtmd/clip.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/mtmd.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/mtmd-helper.cpp
)

target_include_directories(mtmd
  PUBLIC
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/tools/mtmd>
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/common>
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/vendor>
    $<INSTALL_INTERFACE:include>   
)

# CUDA
if(GGML_CUDA)
  add_compile_definitions(GGML_USE_CUDA)
endif()

# export
# file(GLOB COMMON_HEADERS 
#   ${llama_SOURCE_DIR}/ggml/include/*
#   ${llama_SOURCE_DIR}/common/*.h
#   ${llama_SOURCE_DIR}/common/*.hpp
#   ${llama_SOURCE_DIR}/tools/mtmd/*.h
#   ${llama_SOURCE_DIR}/vendor/nlohmann/*.hpp
#   ${llama_SOURCE_DIR}/vendor/minja/*.hpp
# )
install(
  DIRECTORY
    ${llama_SOURCE_DIR}/common/
    ${llama_SOURCE_DIR}/ggml/include/
    ${llama_SOURCE_DIR}/tools/mtmd/
    ${llama_SOURCE_DIR}/vendor/nlohmann/
    ${llama_SOURCE_DIR}/vendor/minja/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

set(INSTALL_TARGETS 
  ggml
  ggml-base
  ggml-cpu
  build_info
  common
  llama
  mtmd
)

if(GGML_CUDA)
  list(APPEND INSTALL_TARGETS ggml-cuda)
endif()

install(
  TARGETS ${INSTALL_TARGETS}
  EXPORT export_llama
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(export_llama HAS_LIBRARY_TARGET)
ament_package()
