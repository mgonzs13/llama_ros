cmake_minimum_required(VERSION 3.8)
project(llama_cpp_vendor)

include(FetchContent)
find_package(ament_cmake REQUIRED)
find_package(OpenSSL QUIET)

FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b8149
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

if(GGML_HIP)
  set_target_properties(
    ggml-hip PROPERTIES
    OUTPUT_NAME "llama_ggml_hip"
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
  ${llama_SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/cogvlm.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/conformer.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/glm4v.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/internvl.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/kimik25.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/kimivl.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/llama4.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/llava.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/minicpmv.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/mobilenetv5.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/nemotron-v2-vl.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/paddleocr.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/pixtral.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/qwen2vl.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/qwen3vl.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/siglip.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/whisper-enc.cpp
  ${llama_SOURCE_DIR}/tools/mtmd/models/youtuvl.cpp
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

if(GGML_CUDA)
  add_compile_definitions(GGML_USE_CUDA)
endif()

if(GGML_HIP)
  add_compile_definitions(GGML_USE_HIP)
endif()

# export
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

install(
  DIRECTORY
    ${llama_SOURCE_DIR}/models/templates/
  DESTINATION share/${PROJECT_NAME}/models/templates
  FILES_MATCHING PATTERN "*.jinja"
)

set(INSTALL_TARGETS 
  ggml
  ggml-base
  ggml-cpu
  build_info
  common
  llama
  mtmd
  cpp-httplib
)

if(GGML_CUDA)
  list(APPEND INSTALL_TARGETS ggml-cuda)
endif()

if(GGML_HIP)
  list(APPEND INSTALL_TARGETS ggml-hip)
endif()

install(
  TARGETS ${INSTALL_TARGETS}
  EXPORT export_llama
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(export_llama HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenSSL)
ament_package()