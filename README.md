# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. By using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/mgonzs13/llama_ros.git
$ cd ~/ros2_ws
$ colcon build
```


## LLMs

 - [LLaMA](https://huggingface.co/TheBloke/LLaMa-7B-GGML)
 - [Alpaca](https://huggingface.co/TheBloke/gpt4-x-alpaca-13B-GGML)
 - [GPT4All](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML)
 - [Vicuna](https://huggingface.co/TheBloke/Vicuna-7B-CoT-GGML)
 - [Vicuan-1.1](https://huggingface.co/TheBloke/vicuna-7B-1.1-GGML)
 - [WizardLM](https://huggingface.co/TheBloke/wizardLM-7B-GGML)
 - [WizardVicuna](https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML)
 - [Airoboros](https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.2-GGML)
 - [Tulu](https://huggingface.co/TheBloke/tulu-7B-GGML)
 - [Robin](https://huggingface.co/TheBloke/robin-7B-v2-GGML)
 - [Gorilla](https://huggingface.co/TheBloke/gorilla-7B-GGML)
 - [Guanaco](https://huggingface.co/TheBloke/guanaco-7B-GGML)
 - [Flan](https://huggingface.co/TheBloke/Flan-OpenLlama-7B-GGML)
 - [Orca](https://huggingface.co/TheBloke/orca_mini_7B-GGML)
 - [LLaMA2](https://huggingface.co/TheBloke/Llama-2-7B-GGML)
 - [Nous Hermes](https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGML)


## Usage

  - Download the models and place them in `~/llama_models` 
  - Rename the bin (check the content of the launch files)
  - Run the launch file of the chosen model


## Demo

```shell
$ ros2 launch llama_bringup wizard-vicuna.launch.py
```

```shell
$ ros2 run llama_ros llama_client_node --ros-args -p prompt:="your prompt"
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->
https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

