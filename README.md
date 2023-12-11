# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. By using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/mgonzs13/llama_ros.git
$ cd ~/ros2_ws
$ colcon build
```

### CUDA

To run llama_ros with CUDA, the following lines in the [CMakeLists.txt](llama_ros/CMakeLists.txt) must be uncommented:

```
option(LLAMA_CUBLAS "llama: use cuBLAS" ON)
add_compile_definitions(GGML_USE_CUBLAS)
```

## Usage

Create and run the launch file:

```python
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch


def generate_launch_description():

    return LaunchDescription([
        create_llama_launch(
            n_ctx=512, # context of the LLM in tokens
            n_batch=8, # batch size in tokens
            n_gpu_layers=0, # layers to load in GPU
            n_threads=4, # threads
            n_predict=512, # max tokens (prompt tokens + predicted tokens

            model_repo="TheBloke/Marcoroni-7B-v3-GGUF", # Hugging Face repo
            model_filename="marcoroni-7b-v3.Q4_K_M.gguf", # model file

            prefix="\n\n### Instruction:\n", # prefix to add at the start of the prompt
            suffix="\n\n### Response:\n", # suffix to add at the end of the prompt
            stop="### Instruction:\n", # stop sequence

            file="alpaca.txt" # initial prompt
        )
    ])
```

Run you launch file:

```shell
$ ros2 launch llama_bringup nous-hermes.launch.py
```

Send an action goal:

```shell
$ ros2 action send_goal /llama/generate_response llama_msgs/action/GenerateResponse "{'prompt': 'What is ROS2?'}"
```

### Lagnchain

There is a [llama_ros integration for langchain](llama_ros/llama_ros/langchain/) based on the [simple_node](https://github.com/uleroboticsgroup/simple_node) pacakge.

## Demo

```shell
$ ros2 launch llama_bringup wizard-vicuna.launch.py
```

```shell
$ ros2 run llama_ros llama_client_node --ros-args -p prompt:="your prompt"
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->

https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

## LLMs

[[](https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/HuggingFaceH4/open_llm_leaderboard.png)](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
