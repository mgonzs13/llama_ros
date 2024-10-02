cmake_minimum_required(VERSION 3.8)
project(llama_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch prompts models
  DESTINATION share/${PROJECT_NAME}/
)

# Python
ament_python_install_package(${PROJECT_NAME})

ament_package()
