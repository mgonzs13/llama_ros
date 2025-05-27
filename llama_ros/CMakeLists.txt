cmake_minimum_required(VERSION 3.8)
project(llama_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(llama_msgs REQUIRED)
find_package(llama_cpp_vendor REQUIRED)
find_package(Threads REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(llama_hfhub_vendor REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# ROS 2 distros
if("$ENV{ROS_DISTRO}" STREQUAL "foxy" OR "$ENV{ROS_DISTRO}" STREQUAL "galactic" OR "$ENV{ROS_DISTRO}" STREQUAL "humble")
  message(STATUS "Using cv_bridge.h")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_BRIDGE_H")
else()
  message(STATUS "Using cv_bridge.hpp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_BRIDGE_HPP")
endif()

# llama nodes
add_executable(llama_node
  src/llama_ros/llama.cpp
  src/llama_utils/logs.cpp
  src/llama_utils/chat_utils.cpp
  src/llama_utils/llama_params.cpp
  src/llama_ros/llama_node.cpp
  src/llama_main.cpp
)
target_link_libraries(llama_node
  llama_cpp_vendor::llama
  llama_cpp_vendor::common
  ${CMAKE_THREAD_LIBS_INIT}
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  rclcpp_lifecycle::rclcpp_lifecycle
  ${llama_msgs_TARGETS}
  ${llama_cpp_vendor_TARGETS}
  ${llama_hfhub_vendor_TARGETS}
  Threads::Threads
)

add_executable(llava_node
  src/llama_ros/llama.cpp
  src/llava_ros/llava.cpp
  src/llama_utils/logs.cpp
  src/llama_utils/chat_utils.cpp
  src/llama_utils/llama_params.cpp
  src/llama_ros/llama_node.cpp
  src/llava_ros/llava_node.cpp
  src/llava_main.cpp
)
target_link_libraries(llava_node
  cv_bridge::cv_bridge
  llama_cpp_vendor::llama
  llama_cpp_vendor::mtmd
  llama_cpp_vendor::common
  ${CMAKE_THREAD_LIBS_INIT}
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  rclcpp_lifecycle::rclcpp_lifecycle
  ${llama_msgs_TARGETS}
  ${llama_cpp_vendor_TARGETS}
  ${llama_hfhub_vendor_TARGETS}
  Threads::Threads
)

ament_export_dependencies(llama_cpp_vendor)

# INSTALL
install(TARGETS
  llama_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  llava_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)

# TEST
if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)
  ament_clang_format(CONFIG_FILE .clang-format)
endif()

ament_python_install_package(${PROJECT_NAME})
ament_package()
