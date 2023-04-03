# llama_ros

This repositiory provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2. By using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/mgonzs13/llama_ros.git
$ cd ~/ros2_ws
$ colcon build
```

## Usage

Download the models, place them in `~/llama_models` and rename the models as follow:

- `llama`: `llama.bin`
- `alpaca`: `alpaca.bin`
- `gpt4all`: `gpt4all.bin`

```shell
$ ros2 launch llama_bringup alpaca.launch.py
```

```shell
$ ros2 run llama_ros llama_client_node
```

https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4
