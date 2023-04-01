# llama_ros

This repositiory provides a set of ROS 2 packages to integrate the [llama.cpp](https://github.com/ggerganov/llama.cpp) library into your ROS 2. Since [llama.cpp](https://github.com/ggerganov/llama.cpp) is an efficient and lightweight C++ tool to use LLM models (`llama`, `alpaca`, `gpt4all`), it can be used by the robotics community easily. By using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 application.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/mgonzs13/llama_ros.git
$ cd ~/ros2_ws
$ colcon build
```

## Usage

Download the models (`llama`, `alpaca`, `gpt4all`) and place them in `~/llama_models`.

```shell
$ ros2 launch llama_bringup alpaca.launch.py
```
