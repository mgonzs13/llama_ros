# llama_ros

Lama.cpp for ROS 2

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
