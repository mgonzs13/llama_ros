cmake_minimum_required(VERSION 3.8)
project(llama_hfhub_vendor)

include(FetchContent)
find_package(ament_cmake REQUIRED)
find_package(CURL REQUIRED)  # Add CURL as a required package

# Declare hfhub from Git repository
FetchContent_Declare(
  hfhub
  GIT_REPOSITORY https://github.com/agonzc34/huggingface-hub-cpp
  GIT_TAG        1.1.0
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(hfhub)

# Export targets and include directories
install(
  TARGETS hfhub
  EXPORT export_hfhub
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

# Export include directories and dependencies (CURL)
ament_export_include_directories(include)
ament_export_dependencies(CURL)  # Export CURL dependency
ament_export_targets(export_hfhub HAS_LIBRARY_TARGET)

ament_package()
