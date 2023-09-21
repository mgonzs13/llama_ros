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

- Download the models and place them in `~/llama_models`
- Rename the bin (check the content of the launch files)
- Run the launch file of the chosen model

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
      <th>Base Model</th>
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
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.4-GGML"
          >7B</a
        >,
        <a href="https://huggingface.co/TheBloke/airoboros-13B-gpt4-1.4-GGML"
          >13B</a
        >
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
        <a href="https://github.com/jondurbin/airoboros">Airoboros-12</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/airoboros-l2-7b-gpt4-2.0-GGML"
          >7B</a
        >,
        <a href="https://huggingface.co/TheBloke/airoboros-l2-13b-gpt4-2.0-GGML"
          >13B</a
        >
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
        <a href="https://crfm.stanford.edu/2023/03/13/alpaca.html">Alpaca</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/gpt4-x-alpaca-13B-GGML">13B</a>
      </td>
      <td align="left">
        <code>### Instruction: prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/alpaca.launch.py">alpaca.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/conceptofmind/Flan-Open-Llama-7b"
          >Flan</a
        >
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Flan-OpenLlama-7B-GGML">7B</a>
      </td>
      <td align="left">
        <code>### Instruction: prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/flan.launch.py">flan.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/artidoro/qlora">Guanaco</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/guanaco-7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/guanaco-13B-GGML">13B</a>
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
      <td align="left">
        <a href="https://github.com/artidoro/qlora">Guanaco-QLoRA</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GGML"
          >7B</a
        >,
        <a href="https://huggingface.co/TheBloke/llama-2-13B-Guanaco-QLoRA-GGML"
          >13B</a
        >
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
      <td align="left">
        <a href="https://shishirpatil.github.io/gorilla/">Gorilla</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/gorilla-7B-GGML">7B</a>
      </td>
      <td align="left">
        <code>###USER: prompt</code><br />
        <code>###ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/gorilla.launch.py">gorilla.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://gpt4all.io/index.html">GPT4All</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/GPT4All-13B-snoozy-GGML"
          >13B</a
        >
      </td>
      <td align="left">
        <code>### Instruction: prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/gpt4all.launch.py">gpt4all.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/NousResearch">Nous-Hermes</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Nous-Hermes-Llama-2-7B-GGML"
          >7B</a
        >,
        <a href="https://huggingface.co/TheBloke/Nous-Hermes-Llama2-GGML"
          >13B</a
        >
      </td>
      <td align="left">
        <code>### Instruction: prompt</code><br />
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
        <a href="https://ai.meta.com/blog/large-language-model-llama-meta-ai/"
          >LLaMA</a
        >
      </td>
      <td align="center">-</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/LLaMa-7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/LLaMa-13B-GGML">13B</a>
      </td>
      <td align="left">-</td>
      <td align="left">
        <a href="llama_bringup/launch/llama_chat.launch.py"
          >llama_chat.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left"><a href="https://ai.meta.com/llama/">LLaMA2</a></td>
      <td align="center">-</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Llama-2-7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/Llama-2-13B-GGML">13B</a>
      </td>
      <td align="left">-</td>
      <td align="left">
        <a href="llama_bringup/launch/llama2.launch.py">llama2.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/psmathur/orca_mini_v3_7b"
          >Orca</a
        >
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/orca_mini_v3_7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML">13B</a>
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
        <a href="https://github.com/OptimalScale/LMFlow/">Robin</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/robin-7B-v2-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/Robin-7B-v2-SuperHOT-8K-GGML"
          >7B-8K</a
        >, <a href="https://huggingface.co/TheBloke/robin-13B-v2-GGML">13B</a>,
        <a href="https://huggingface.co/TheBloke/Robin-13B-v2-SuperHOT-8K-GGML"
          >13B-8K</a
        >
      </td>
      <td align="left">
        <code>###Human: prompt</code><br />
        <code>###Assistant:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/robin.launch.py">robin.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://huggingface.co/stabilityai">StableBeluga</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/StableBeluga-7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/StableBeluga-13B-GGML">13B</a>
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
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/tulu-7B-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/tulu-7B-SuperHOT-8K-GGML"
          >7B-8K</a
        >, <a href="https://huggingface.co/TheBloke/tulu-13B-GGML">13B</a>,
        <a href="https://huggingface.co/TheBloke/tulu-13B-SuperHOT-8K-GGML"
          >13B-8K</a
        >
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
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/Vicuna-7B-CoT-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/Vicuna-13B-CoT-GGML">13B</a>
      </td>
      <td align="left">
        <code>### Human: prompt</code><br />
        <code>### Assistant:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/vicuna.launch.py">vicuna.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://lmsys.org/blog/2023-03-30-vicuna/">Vicuna-1</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGML">7B</a>,
        <a href="https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGML">13B</a>
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/vicuna_1.launch.py">vicuna_1.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/nlpxucan/WizardLM">WizardLM</a>
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a href="https://huggingface.co/TheBloke/wizardLM-7B-GGML">7B</a>
      </td>
      <td align="left">
        <code>### Instruction: prompt</code><br />
        <code>### Response:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/wizardlm.launch.py">wizardlm.launch.py</a>
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/nlpxucan/WizardLM">WizardLM-1</a>
      </td>
      <td align="center">LLaMA2</td>
      <td align="left">
        <a
          href="https://huggingface.co/TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGML"
          >13B</a
        >
      </td>
      <td align="left">
        <code>USER: prompt</code><br />
        <code>ASSISTANT:</code>
      </td>
      <td align="left">
        <a href="llama_bringup/launch/wizardlm_1.launch.py"
          >wizardlm_1.launch.py</a
        >
      </td>
    </tr>
    <tr>
      <td align="left">
        <a href="https://github.com/melodysdreamj/WizardVicunaLM"
          >WizardVicuna</a
        >
      </td>
      <td align="center">LLaMA</td>
      <td align="left">
        <a
          href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGML"
          >7B</a
        >,
        <a
          href="https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML"
          >13B</a
        >
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
  </tbody>
</table>
