cmake_minimum_required(VERSION 3.8)
project(llama_cpp_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)
find_package(ament_cmake REQUIRED)

FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b4056
)

option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)

FetchContent_MakeAvailable(llama)

# ggml
set_target_properties(
  ggml PROPERTIES
  OUTPUT_NAME "llama_ggml"
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>"
  CXX_STANDARD 11
)

# llama
set_target_properties(
  build_info llama common PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_SOURCE_DIR}/include;${llama_SOURCE_DIR}/src;${llama_SOURCE_DIR}/common;${llama_SOURCE_DIR}/>"
  CXX_STANDARD 11
)

# llava
add_library(llava
  ${llama_SOURCE_DIR}/examples/llava/clip.cpp
  ${llama_SOURCE_DIR}/examples/llava/llava.cpp
)

target_include_directories(llava
  PUBLIC
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/examples/llava;${llama_SOURCE_DIR}/common;${llama_SOURCE_DIR}/include;${llama_SOURCE_DIR}/ggml/include>
    $<INSTALL_INTERFACE:include>
)

# CUDA
if(GGML_CUDA)
  add_compile_definitions(GGML_USE_CUDA)
endif()

# export
file(GLOB COMMON_HEADERS 
  ${llama_SOURCE_DIR}/ggml/include/*
  ${llama_SOURCE_DIR}/common/*.h
  ${llama_SOURCE_DIR}/common/*.hpp
  ${llama_SOURCE_DIR}/examples/llava/*.h
)
install(
  FILES ${COMMON_HEADERS}
  DESTINATION include
)

install(
  TARGETS ggml build_info llama llava common
  EXPORT export_llama
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(export_llama HAS_LIBRARY_TARGET)
ament_package()
