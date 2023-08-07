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

| LLM | Base Model | Sizes | Prompt Template | Launch File Example |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) | - | [7B](https://huggingface.co/TheBloke/LLaMa-7B-GGML), [13B](https://huggingface.co/TheBloke/LLaMa-13B-GGML) | - | [llama_chat.launch.py](llama_bringup/launch/llama_chat.launch.py) |
| [LLaMA2](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) | - | [7B](https://huggingface.co/TheBloke/Llama-2-7B-GGML), [13B](https://huggingface.co/TheBloke/Llama-2-13B-GGML) | - | [llama2.launch.py](llama_bringup/launch/llama2.launch.py) |
| [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) | LLaMA | [13B](https://huggingface.co/TheBloke/gpt4-x-alpaca-13B-GGML) | \#\#\# Instruction: prompt<br />\#\#\# Response: | [alpaca.launch.py](llama_bringup/launch/alpaca.launch.py) |
| [GPT4All](https://gpt4all.io/index.html) | LLaMA | [13B](https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML) | \#\#\# Instruction: prompt<br />\#\#\# Response: | [gpt4all.launch.py](llama_bringup/launch/gpt4all.launch.py) |
| [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) | LLaMA | [7B](https://huggingface.co/TheBloke/Vicuna-7B-CoT-GGML), [13B](https://huggingface.co/TheBloke/Vicuna-13B-CoT-GGML) | \#\#\# Human: prompt<br />\#\#\# Assistant: | [vicuna.launch.py](llama_bringup/launch/vicuna.launch.py) |
| [Vicuna-1](https://lmsys.org/blog/2023-03-30-vicuna/) | LLaMA2 | [7B](https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGML), [13B](https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGML) | USER: prompt<br />ASSISTANT: | [vicuna_1.launch.py](llama_bringup/launch/vicuna_1.launch.py) |
| [WizardLM](https://github.com/nlpxucan/WizardLM) | LLaMA | [7B](https://huggingface.co/TheBloke/wizardLM-7B-GGML) | \#\#\# Instruction: prompt<br />\#\#\# Response: | [wizardlm.launch.py](llama_bringup/launch/wizardlm.launch.py) |
| [WizardLM-1](https://github.com/nlpxucan/WizardLM) | LLaMA2 | [13B](https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGML) | USER: prompt<br />ASSISTANT: | [wizardlm_1.launch.py](llama_bringup/launch/wizardlm_1.launch.py) |
| [WizardVicuna](https://github.com/melodysdreamj/WizardVicunaLM) | LLaMA | [7B](https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML), [13B](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML) | USER: prompt<br />ASSISTANT: | [wizardlm-vicuna.launch.py](llama_bringup/launch/wizard-vicuna.launch.py) |
[Airoboros](https://github.com/jondurbin/airoboros) | LLaMA | [7B](https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.4-GGML), [13B](https://huggingface.co/TheBloke/airoboros-13B-gpt4-1.4-GGML) | USER: prompt<br />ASSISTANT: | [airoboros.launch.py](llama_bringup/launch/airoboros.launch.py) |
[Airoboros-12](https://github.com/jondurbin/airoboros) | LLaMA2 | [7B](https://huggingface.co/TheBloke/airoboros-l2-7b-gpt4-2.0-GGML), [13B](https://huggingface.co/TheBloke/airoboros-l2-13b-gpt4-2.0-GGML) | USER: prompt<br />ASSISTANT: | [airoboros.launch.py](llama_bringup/launch/airoboros.launch.py) |
[Tulu](https://github.com/allenai/open-instruct) | LLaMA | [7B](https://huggingface.co/TheBloke/tulu-7B-GGML), [7B-8K](https://huggingface.co/TheBloke/tulu-7B-SuperHOT-8K-GGML), [13B](https://huggingface.co/TheBloke/tulu-13B-GGML), [13B-8K](https://huggingface.co/TheBloke/tulu-13B-SuperHOT-8K-GGML) | <\|user\|><br />prompt<br /><\|assistant\|> | [tulu.launch.py](llama_bringup/launch/tulu.launch.py) |
[Robin](https://github.com/OptimalScale/LMFlow/) | LLaMA | [7B](https://huggingface.co/TheBloke/robin-7B-v2-GGML), [7B-8K](https://huggingface.co/TheBloke/Robin-7B-v2-SuperHOT-8K-GGML), [13B](https://huggingface.co/TheBloke/robin-13B-v2-GGML), [13B-8K](https://huggingface.co/TheBloke/Robin-13B-v2-SuperHOT-8K-GGML) | \#\#\#Human: prompt<br />\#\#\#Assistant: | [robin.launch.py](llama_bringup/launch/robin.launch.py) |
[Gorilla](https://shishirpatil.github.io/gorilla/) | LLaMA | [7B](https://huggingface.co/TheBloke/gorilla-7B-GGML) | \#\#\#USER: prompt<br />\#\#\#ASSISTANT: | [gorilla.launch.py](llama_bringup/launch/gorilla.launch.py) |
[Guanaco](https://github.com/artidoro/qlora) | LLaMA | [7B](https://huggingface.co/TheBloke/guanaco-7B-GGML), [13B](https://huggingface.co/TheBloke/guanaco-13B-GGML) | \#\#\# Human: prompt<br />\#\#\# Assistant: | [guanaco.launch.py](llama_bringup/launch/guanaco.launch.py) |
[Flan](https://huggingface.co/conceptofmind/Flan-Open-Llama-7b) | LLaMA | [7B](https://huggingface.co/TheBloke/Flan-OpenLlama-7B-GGML) | \#\#\# Instruction: prompt<br />\#\#\# Response: | [flan.launch.py](llama_bringup/launch/flan.launch.py) |
[Orca](https://huggingface.co/conceptofmind/Flan-Open-Llama-7b) | LLaMA | [7B](https://huggingface.co/TheBloke/orca_mini_v2_7B-GGML), [13B](https://huggingface.co/TheBloke/orca_mini_v2_13B-GGML) | \#\#\# System:<br />system prompt<br /><br />\#\#\# User:<br />prompt<br /><br />\#\#\# Input:<br />input<br /><br />\#\#\# Response: | [orca.launch.py](llama_bringup/launch/orca.launch.py) |
[Guanaco](https://github.com/artidoro/qlora) | LLaMA2 | [7B](https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GGML), [13B](https://huggingface.co/TheBloke/llama-2-13B-Guanaco-QLoRA-GGML) | \#\#\# Human: prompt<br />\#\#\# Assistant: | [guanaco.launch.py](llama_bringup/launch/guanaco.launch.py) |
[Nous-Hermes](https://huggingface.co/NousResearch) | LLaMA2 | [7B](https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGML), [13B](https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGML) | \#\#\# Instruction: prompt<br />\#\#\# Response: | [nous-hermes.launch.py](llama_bringup/launch/nous-hermes.launch.py) |
[StableBeluga](https://huggingface.co/stabilityai) | LLaMA2 | [7B](https://huggingface.co/TheBloke/StableBeluga-7B-GGML), [13B](https://huggingface.co/TheBloke/StableBeluga-13B-GGML) | \#\#\# System:<br />system prompt<br /><br />\#\#\# User:<br />prompt<br /><br />\#\#\# Assistant: | [stablebeluga.launch.py](llama_bringup/launch/stablebeluga.launch.py) |


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

