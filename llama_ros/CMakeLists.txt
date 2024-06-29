cmake_minimum_required(VERSION 3.8)
project(llama_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA
# set(GGML_CUDA ON)
if (GGML_CUDA)
  add_compile_definitions(GGML_USE_CUDA)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(llama_msgs REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(llama_cpp)
add_subdirectory(llama_cpp/examples/llava)


add_executable(llama_node
  src/llama_ros/llama.cpp 
  src/llama_utils/gpt_params.cpp 
  src/llama_ros/llama_node.cpp 
  src/llama_main.cpp
)
target_link_libraries(llama_node PRIVATE common ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(llama_node PRIVATE llama)
ament_target_dependencies(llama_node PUBLIC rclcpp rclcpp_action llama_msgs)

add_executable(llava_node
  src/llama_ros/llama.cpp 
  src/llava_ros/llava.cpp 
  src/llama_utils/gpt_params.cpp 
  src/llama_ros/llama_node.cpp 
  src/llava_ros/llava_node.cpp 
  src/llava_main.cpp
)
target_link_libraries(llava_node PRIVATE common ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(llava_node PRIVATE PRIVATE llava llama)
ament_target_dependencies(llava_node PUBLIC rclcpp rclcpp_action llama_msgs cv_bridge)

# INSTALL
install(TARGETS
  llama_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  llava_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  llama_ros/llama_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llama_demo_node
)

install(PROGRAMS
  llama_ros/llava_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llava_demo_node
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
