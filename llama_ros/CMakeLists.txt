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
if(NOT DEFINED ENV{ROS_DISTRO})
  message(FATAL_ERROR "ROS_DISTRO is not defined." )
endif()
if("$ENV{ROS_DISTRO}" STREQUAL "foxy")
  message(STATUS "Build for ROS 2 Foxy")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFOXY")
elseif("$ENV{ROS_DISTRO}" STREQUAL "humble")
  message(STATUS "Build for ROS 2 Humble")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHUMBLE")
elseif("$ENV{ROS_DISTRO}" STREQUAL "iron")
  message(STATUS "Build for ROS 2 Iron")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIRON")
elseif("$ENV{ROS_DISTRO}" STREQUAL "jazzy")
  message(STATUS "Build for ROS 2 Jazzy")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJAZZY")
else()
  message(FATAL_ERROR "Unsupported ROS Distribution: " "$ENV{ROS_DISTRO}")
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
  Threads::Threads
)
ament_target_dependencies(llama_node
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  llama_msgs
  llama_cpp_vendor
  llama_hfhub_vendor
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
  llama_cpp_vendor::llama
  llama_cpp_vendor::llava
  llama_cpp_vendor::common
  ${CMAKE_THREAD_LIBS_INIT}
  Threads::Threads
)
ament_target_dependencies(llava_node
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  llama_msgs
  llama_cpp_vendor
  cv_bridge
  llama_hfhub_vendor
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
