cmake_minimum_required(VERSION 3.8)
project(llama_hfhub_vendor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)
find_package(ament_cmake REQUIRED)

FetchContent_Declare(
  hfhub
  GIT_REPOSITORY https://github.com/mgonzs13/huggingface-hub-cpp
  GIT_TAG        ce23f9c99797bc9d4c9632f1fb71f06e7c8ec6a8
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(hfhub)

install(
  TARGETS hfhub
  EXPORT export_hfhub
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(export_hfhub HAS_LIBRARY_TARGET)
ament_package()
