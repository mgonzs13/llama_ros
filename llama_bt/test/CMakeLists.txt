include_directories(.)

add_subdirectory(action)

# Test register
ament_add_gtest(test_register
  test_register.cpp
)
target_link_libraries(test_register
  rclcpp::rclcpp
)
ament_target_dependencies(test_register
  behaviortree_cpp
)
