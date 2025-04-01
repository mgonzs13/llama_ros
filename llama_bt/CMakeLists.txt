cmake_minimum_required(VERSION 3.5)
project(llama_bt)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE
    STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  if("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
  else()
    message(FATAL_ERROR "cxx_std_17 could not be found.")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wdeprecated -fPIC -Wshadow -Wnull-dereference)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wnon-virtual-dtor>")
endif()

if(NOT DEFINED ENV{ROS_DISTRO})
  message(FATAL_ERROR "ROS_DISTRO is not defined.")
endif()

if("$ENV{ROS_DISTRO}" STREQUAL "foxy" OR "$ENV{ROS_DISTRO}" STREQUAL "galactic" OR "$ENV{ROS_DISTRO}" STREQUAL "humble")
  message(STATUS "Using BehaviorTree.CPP V3")
  set(BT_LIB_NAME behaviortree_cpp_v3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBTV3")
elseif("$ENV{ROS_DISTRO}" STREQUAL "iron" OR "$ENV{ROS_DISTRO}" STREQUAL "jazzy" OR "$ENV{ROS_DISTRO}" STREQUAL "rolling")
  message(STATUS "Using BehaviorTree.CPP V4")
  set(BT_LIB_NAME behaviortree_cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBTV4")
else()
  message(FATAL_ERROR "Unsupported ROS Distribution: " "$ENV{ROS_DISTRO}")
endif()

option(COVERAGE_ENABLED "Enable code coverage" FALSE)

if(COVERAGE_ENABLED)
  add_compile_options(--coverage)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

# Defaults for Microsoft C++ compiler
if(MSVC)
  # https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

  # Enable Math Constants
  # https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=vs-2019
  add_compile_definitions(
    _USE_MATH_DEFINES
  )
endif()

# ###############################################
# # Find  dependencies                         ##
# ###############################################
# # Find ament macros and libraries
find_package(ament_cmake REQUIRED)
find_package(${BT_LIB_NAME} REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(llama_msgs REQUIRED)
find_package(llama_ros REQUIRED)

# ##########
# # Build ##
# ##########
add_library(llama_generate_response_action_bt_node SHARED src/action/generate_response_action.cpp)
list(APPEND plugin_libs llama_generate_response_action_bt_node)

add_library(llama_generate_chat_completions_action_bt_node SHARED src/action/generate_chat_completions_action.cpp)
list(APPEND plugin_libs llama_generate_chat_completions_action_bt_node)

foreach(bt_plugin ${plugin_libs})
  target_include_directories(${bt_plugin} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
  )
  target_link_libraries(${bt_plugin} PUBLIC
    rclcpp::rclcpp
    rclcpp_action::rclcpp_action
    rclcpp_lifecycle::rclcpp_lifecycle
    ${llama_msgs_TARGETS}
    ${llama_ros_TARGETS}
  )
  ament_target_dependencies(${bt_plugin} PUBLIC ${BT_LIB_NAME}) # TODO(ajtudela): Fix this in jazzy
  target_compile_definitions(${bt_plugin} PRIVATE BT_PLUGIN_EXPORT)
endforeach()

# ############
# # Install ##
# ############
install(TARGETS ${plugin_libs}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY test/utils/
  DESTINATION include/${PROJECT_NAME}/utils/
)

install(FILES llama_tree_nodes.xml DESTINATION share/${PROJECT_NAME})

# ###########
# Testing  ##
# ###########
if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)

  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  find_package(ament_cmake_gtest REQUIRED)

  add_subdirectory(test)
endif()

# ##################################
# # ament specific configuration  ##
# ##################################
ament_export_include_directories(include/${PROJECT_NAME})
ament_export_libraries()
ament_export_dependencies(
  ${BT_LIB_NAME}
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  llama_msgs
  llama_ros
)
ament_export_targets()
ament_package()