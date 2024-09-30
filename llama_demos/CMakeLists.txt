cmake_minimum_required(VERSION 3.8)
project(llama_demos)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(PROGRAMS
  llama_demos/llama_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llama_demo_node
)

install(PROGRAMS
  llama_demos/llama_rerank_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llama_rerank_demo_node
)

install(PROGRAMS
  llama_demos/llava_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME llava_demo_node
)

install(PROGRAMS
  llama_demos/chatllama_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME chatllama_demo_node
)

ament_python_install_package(${PROJECT_NAME})
ament_package()
