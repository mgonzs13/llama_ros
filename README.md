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

            model_repo="TheBloke/Nous-Hermes-Llama-2-7B-GGUF", # Hugging Face repo
            model_filename="nous-hermes-llama-2-7b.Q4_K_M.gguf", # model file

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

<table>
  <thead>
    <tr>
      <th>LLM</th>
      <th>Sizes</th>
      <th>Prompt Template</th>
      <th>Launch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">
        <a href="https://github.com/jondurbin/airoboros">Airoboros</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/airoboros-l2-7B-2.2.1-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/airoboros-l2-13B-2.2.1-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/airoboros-c34b-2.2.1-GGUF">34B</a>,
        <a href="https://huggingface.co/TheBloke/Airoboros-L2-70b-2.2.1-GGUF">70B</a>
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/airoboros.launch.py"
          >airoboros.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/ehartford/dolphin-2.1-mistral-7b">Dolphin-Mistral</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF">7B</a>
      </td>
      <td align="left">
        <code><|im_start|>system</code><br />
        <code>system prompt<|im_end|></code><br />
        <code><|im_end|>user</code><br />
        <code>promtp<|im_end|></code><br />
        <code><|im_start|>assistant</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/dolphin-mistral.launch.py"
          >dolphin-mistral.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/artidoro/qlora">Guanaco-QLoRA</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/llama-2-13B-Guanaco-QLoRA-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/llama-2-70B-Guanaco-QLoRA-GGUF">70B</a>
      </td>
      <td align="left">
        <code>### Human: prompt</code><br />
        <code>### Assistant:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/guanaco.launch.py">guanaco.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left"><a href="https://ai.meta.com/llama/">LLaMA2</a></td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Llama-2-7B-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/Llama-2-13B-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/Llama-2-70B-GGUF">70B</a>
      </td>
      <td align="left">-</td>
      <td align="left">
        <a href="llama_bringup/launch/llama2.launch.py">llama2.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/AIDC-ai-business">Marcoroni</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Marcoroni-7b-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/Marcoroni-13b-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/Marcoroni-70b-GGUF">70B</a>
      </td>
      <td align="left">
        <code>### Instruction:</code><br />
        <code>prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/marcoroni.launch.py"
          >marcoroni.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://mistral.ai/news/announcing-mistral-7b/">Mistral</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF">7B</a>
      </td>
      <td align="left">
        <code>[INST] {prompt} [/INST]</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/mistral.launch.py"
          >mistral.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/NousResearch">Nous-Hermes</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/Nous-Hermes-Llama2-70B-GGUF">70B</a>
      </td>
      <td align="left">
        <code>### Instruction:</code><br />
        <code>prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/nous-hermes.launch.py"
          >nous-hermes.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/psmathur/orca_mini_v3_7b"
          >Orca</a
        >
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/orca_mini_v3_7B-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/orca_mini_v3_13B-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/orca_mini_v3_70B-GGUF">70B</a>
      </td>
      <td align="left">
        <code>### System:</code><br />
        <code>system prompt</code><br /><br />
        <code>### User:</code><br />
        <code>prompt</code><br /><br />
        <code>###Input:</code><br />
        <code>input</code><br /><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/orca.launch.py">orca.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/stabilityai">StableBeluga</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/StableBeluga-7B-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/StableBeluga-13B-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/StableBeluga-70B-GGUF">70B</a>
      </td>
      <td align="left">
        <code>### System:</code><br />
        <code>system prompt</code><br /><br />
        <code>### User:</code><br />
        <code>prompt</code><br /><br />
        <code>###Assistant:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/stablebeluga.launch.py"
          >stablebeluga.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/allenai/open-instruct">Tulu</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/tulu-7B-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/tulu-13B-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/tulu-30B-GGUF">30B</a>,
      </td>
      <td align="left">
        <code>&lt;|user|&gt;</code><br />
        <code>prompt</code><br />
        <code>&lt;|assistant|&gt;</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/tulu.launch.py">tulu.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://lmsys.org/blog/2023-03-30-vicuna/">Vicuna</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/vicuna-7B-v1.5-16K-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/vicuna-13B-v1.5-16K-GGUF">13B</a>
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/vicuna.launch.py">vicuna.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/melodysdreamj/WizardVicunaLM"
          >WizardVicuna</a
        >
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF">13B</a>,
        <a href="https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF">30B</a>
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/wizard-vicuna.launch.py"
          >wizardlm-vicuna.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/nlpxucan/WizardLM">WizardLM</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/WizardLM-7B-V1.0-Uncensored-GGUF">7B</a>,
        <a href="https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF">13B</a>
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/wizardlm.launch.py"
          >wizardlm.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha">Zephyr</a>
      </td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF">7B</a>
      </td>
      <td align="left">
        <code><|system|></code><br />
        <code>system prompt&lt;s&gt;</code><br />
        <code><|user|></code><br />
        <code>prompt&lt;s&gt;</code><br />
        <code><|assistant|></code><br />
      </td>
      <td align="left">
        <a href="llama_bringup/launch/zephyr.launch.py"
          >zephyr.launch.py</a
        >
      </td>
    </tr>
  </tbody>
</table>
