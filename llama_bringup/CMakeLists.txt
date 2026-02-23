cmake_minimum_required(VERSION 3.8)
project(llama_bringup)

# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch models
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
