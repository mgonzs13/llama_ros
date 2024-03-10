cmake_minimum_required(VERSION 3.8)
project(llama_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# cuBLAS
# option(LLAMA_CUBLAS "llama: use cuBLAS" ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(llama_msgs REQUIRED)

INCLUDE_DIRECTORIES(
  include
)

add_subdirectory(llama_cpp)

add_executable(llama_node src/llama.cpp src/schema_converter.cpp src/llama_node.cpp src/llama_main.cpp)
target_link_libraries(llama_node llama common)
ament_target_dependencies(llama_node rclcpp rclcpp_action llama_msgs)

# INSTALL
install(TARGETS
  llama_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  llama_ros/llama_client_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llama_client_node
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
