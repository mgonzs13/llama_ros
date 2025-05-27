# Test for the generate response action
find_package(ament_index_cpp REQUIRED)

# Test for the generate response action
ament_add_gtest(test_generate_response_action
  test_generate_response_action.cpp
)
target_link_libraries(test_generate_response_action
  ament_index_cpp::ament_index_cpp
  llama_generate_response_action_bt_node
)

# Test for the generate chat action
ament_add_gtest(test_generate_chat_action
  test_generate_chat_action.cpp
)

target_link_libraries(test_generate_chat_action
  ament_index_cpp::ament_index_cpp
  llama_generate_chat_completions_action_bt_node
)
