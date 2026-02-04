cmake_minimum_required(VERSION 3.8)
project(llama_demos)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

set(DEMO_NODES
  llama_demo_node
  llama_embeddings_demo_node
  llama_rerank_demo_node
  llama_rag_demo_node
  llava_demo_node
  mtmd_audio_demo_node
  chatllama_demo_node
  chatllama_audio_demo_node
  chatllama_multi_audio_demo_node
  chatllama_multi_image_demo_node
  chatllama_multi_image_user_demo_node
  chatllama_structured_demo_node
  chatllama_tools_demo_node
  chatllama_streaming_tools_demo_node
  chatllama_langgraph_demo_node
  chatllama_reasoning_demo_node
  chatllama_reasoning_tools_demo_node
  chatllama_pddl_demo_node
)

foreach(NODE ${DEMO_NODES})
  install(PROGRAMS
    llama_demos/${NODE}.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME ${NODE}
  )
endforeach()

ament_python_install_package(${PROJECT_NAME})
ament_package()
