# llama_ros

This repository provides a set of ROS 2 packages to integrate [llama.cpp](https://github.com/ggerganov/llama.cpp) into ROS 2. Using the llama_ros packages, you can easily incorporate the powerful optimization capabilities of [llama.cpp](https://github.com/ggerganov/llama.cpp) into your ROS 2 projects by running [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)-based [LLMs](https://huggingface.co/models?sort=trending&search=gguf+7b) and [VLMs](https://huggingface.co/models?sort=trending&search=gguf+llava). You can also use features from llama.cpp such as [GBNF grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) and modify LoRAs in real-time.

<div align="center">

[![License: MIT](https://img.shields.io/badge/GitHub-MIT-informational)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/llama_ros.svg?branch=main)](https://github.com/mgonzs13/llama_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/llama_ros)](https://github.com/mgonzs13/llama_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/llama_ros.svg)](https://github.com/mgonzs13/llama_ros/graphs/contributors) [![Python Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/python-formatter.yml?branch=main) [![C++ Formatter Check](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/cpp-formatter.yml?branch=main) [![Doxygen Deployment](https://github.com/mgonzs13/llama_ros/actions/workflows/doxygen-deployment.yml/badge.svg)](https://mgonzs13.github.io/llama_ros/latest)

| ROS 2 Distro |                          Branch                           |                                                                                                     Build status                                                                                                     |                                                                Docker Image                                                                |
| :----------: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
|  **Humble**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |  [![Humble Build](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/humble-build-test.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=humble)  |
|   **Iron**   | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |     [![Iron Build](https://github.com/mgonzs13/llama_ros/actions/workflows/iron-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/iron-build-test.yml?branch=main)      |    [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-iron-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=iron)    |
|  **Jazzy**   | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |    [![Jazzy Build](https://github.com/mgonzs13/llama_ros/actions/workflows/jazzy-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/jazzy-build-test.yml?branch=main)    |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-jazzy-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=jazzy)   |
|  **Kilted**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) |  [![Kilted Build](https://github.com/mgonzs13/llama_ros/actions/workflows/kilted-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/kilted-build-test.yml?branch=main)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-kilted-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=kilted)  |
| **Rolling**  | [`main`](https://github.com/mgonzs13/llama_ros/tree/main) | [![Rolling Build](https://github.com/mgonzs13/llama_ros/actions/workflows/rolling-build-test.yml/badge.svg?branch=main)](https://github.com/mgonzs13/llama_ros/actions/workflows/rolling-build-test.yml?branch=main) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-rolling-blue)](https://hub.docker.com/r/mgons/llama_ros/tags?name=rolling) |

</div>

## Table of Contents

1. [Related Projects](#related-projects)
2. [Installation](#installation)
3. [Docker](#docker)
4. [Usage](#usage)
   - [llama_cli](#llama_cli)
   - [Launch Files](#launch-files)
   - [ROS 2 Parameters](#ros-2-parameters)
   - [Speculative Decoding](#speculative-decoding-speculative)
   - [LoRA Adapters](#lora-adapters)
   - [ROS 2 Clients](#ros-2-clients)
   - [LangChain](#langchain)
5. [Demos](#demos)

## Related Projects

- [chatbot_ros](https://github.com/mgonzs13/chatbot_ros) &rarr; This chatbot, integrated into ROS 2, uses [whisper_ros](https://github.com/mgonzs13/whisper_ros/tree/main), to listen to people speech; and llama_ros, to generate responses. The chatbot is controlled by a state machine created with [YASMIN](https://github.com/uleroboticsgroup/yasmin).
- [explainable_ros](https://github.com/Dsobh/explainable_ROS) &rarr; A ROS 2 tool to explain the behavior of a robot. Using the integration of LangChain, logs are stored in a vector database. Then, RAG is applied to retrieve relevant logs for user questions answered with llama_ros.

## Installation

To run llama_ros with CUDA, first, you must install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Then, you can compile llama_ros with `--cmake-args -DGGML_CUDA=ON` to enable CUDA support.

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/llama_ros.git
pip3 install -r llama_ros/requirements.txt
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args -DGGML_CUDA=ON # add this for CUDA
```

To run the tests:

```shell
colcon test --executor sequential --packages-select llama_ros llama_bt
colcon test-result --verbose
```

## Docker

Build the llama_ros docker or download an image from [DockerHub](https://hub.docker.com/repository/docker/mgons/llama_ros). You can choose to build llama_ros with CUDA (`USE_CUDA`) and choose the CUDA version (`CUDA_VERSION`). Remember that you have to use `DOCKER_BUILDKIT=0` to compile llama_ros with CUDA when building the image.

<!-- To build using CUDA you have to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [configure the default runtime to NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.12.1/user-guide.html#daemon-configuration-file). -->

```shell
DOCKER_BUILDKIT=0 docker build -t llama_ros --build-arg USE_CUDA=1 --build-arg CUDA_VERSION=12-6 .
```

Run the docker container. If you want to use CUDA, you have to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all`.

```shell
docker run -it --rm --gpus all llama_ros
```

## Usage

### llama_cli

Commands are included in llama_ros to speed up the test of GGUF-based LLMs within the ROS 2 ecosystem. This way, the following commands are integrating into the ROS 2 commands:

#### launch

Using this command launch a LLM from a YAML file. The configuration of the YAML is used to launch the LLM in the same way as using a regular launch file. Here is an example of how to use it:

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/StableLM-Zephyr.yaml
```

#### prompt

Using this command send a prompt to a launched LLM. The command uses a string, which is the prompt and has the following arguments:

- (`-r`, `--reset`): Whether to reset the LLM before prompting
- (`-t`, `--temp`): The temperature value
- (`--image-url`): Image url to sent to a VLM

Here is an example of how to use it:

```shell
ros2 llama prompt "Do you know ROS 2?" -t 0.0
```

### Launch Files

First of all, you need to create a launch file to use llama_ros or llava_ros. This launch file will contain the main parameters to download the model from HuggingFace and configure it. Take a look at the following examples and the [predefined launch files](llama_bringup/launch).

#### llama_ros (Python Launch)

<details>
<summary>Click to expand</summary>

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llama_node",
            name="llama_node",
            namespace="llama",
            parameters=[{
                "n_ctx": 2048,
                "n_batch": 8,
                "n_predict": 2048,
                "n_gpu_layers": 0,
                "cpu.n_threads": 1,
                "model.repo": "TheBloke/Marcoroni-7B-v3-GGUF",
                "model.filename": "marcoroni-7b-v3.Q4_K_M.gguf",
                "system_prompt_type": "Alpaca",
            }],
        )
    ])
```

```shell
ros2 launch llama_bringup marcoroni.launch.py
```

</details>

#### llama_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 2048
    n_batch: 8
    n_predict: 2048
    n_gpu_layers: 0
    cpu:
      n_threads: 1
    model:
      repo: "cstr/Spaetzle-v60-7b-GGUF"
      filename: "Spaetzle-v60-7b-q4-k-m.gguf"
    system_prompt_type: "Alpaca"
```

```python
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llama_node",
            name="llama_node",
            namespace="llama",
            parameters=[os.path.join(
                get_package_share_directory("llama_bringup"),
                "models", "Spaetzle.yaml")],
        )
    ])
```

```shell
ros2 launch llama_bringup spaetzle.launch.py
```

</details>

#### llama_ros (YAML Config + model shards)

<details>
<summary>Click to expand</summary>

```yaml
n_ctx: 2048 # context of the LLM in tokens
n_batch: 8 # batch size in tokens
n_predict: 2048 # max tokens, -1 == inf
n_gpu_layers: 0 # layers to load in GPU
cpu:
  n_threads: 1 # threads

model:
  repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" # Hugging Face repo
  filename: "qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf" # model shard file in repo

system_prompt_type: "ChatML" # system prompt type
```

```shell
ros2 llama launch Qwen2.yaml
```

</details>

#### llama_ros (Speculative Decoding)

<details>
<summary>Click to expand</summary>

[Speculative decoding](https://arxiv.org/abs/2302.01318) uses a smaller draft model to predict multiple tokens ahead, then verifies them in parallel with the larger target model. This can significantly speed up text generation when using a small draft model from the same model family. Note that speculative decoding requires `n_parallel: 1`.

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 2048
    n_predict: 2048
    n_gpu_layers: -1
    n_parallel: 1
    cpu:
      n_threads: -1
    model:
      repo: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
      filename: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    speculative:
      type: draft
      n_max: 16
      n_min: 0
      p_min: 0.75
      n_gpu_layers: -1
      model:
        repo: lmstudio-community/Llama-3.2-1B-Instruct-GGUF
        filename: Llama-3.2-1B-Instruct-Q4_K_M.gguf
    system_prompt_type: Llama-3
```

```shell
ros2 launch llama_bringup llama-3-speculative.launch.py
```

</details>

#### llava_ros (Python Launch)

<details>
<summary>Click to expand</summary>

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llava_node",
            name="llava_node",
            namespace="llama",
            parameters=[{
                "n_ctx": 8192,
                "n_batch": 512,
                "n_gpu_layers": 33,
                "cpu.n_threads": 1,
                "n_predict": 8192,
                "model.repo": "cjpais/llava-1.6-mistral-7b-gguf",
                "model.filename": "llava-v1.6-mistral-7b.Q4_K_M.gguf",
                "mmproj.repo": "cjpais/llava-1.6-mistral-7b-gguf",
                "mmproj.filename": "mmproj-model-f16.gguf",
                "system_prompt_type": "Mistral",
            }],
        )
    ])
```

```shell
ros2 launch llama_bringup llava.launch.py
```

</details>

#### llava_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: 33
    cpu:
      n_threads: 1
    model:
      repo: "cjpais/llava-1.6-mistral-7b-gguf"
      filename: "llava-v1.6-mistral-7b.Q4_K_M.gguf"
    mmproj:
      repo: "cjpais/llava-1.6-mistral-7b-gguf"
      filename: "mmproj-model-f16.gguf"
    system_prompt_type: "Mistral"
```

```python
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llava_node",
            name="llava_node",
            namespace="llama",
            parameters=[os.path.join(
                get_package_share_directory("llama_bringup"),
                "models", "llava-mistral.yaml")],
        )
    ])
```

```shell
ros2 launch llama_bringup llava.launch.py
```

</details>

#### llava_ros (Audio)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: 29
    cpu:
      n_threads: -1
    model:
      repo: "mradermacher/Qwen2-Audio-7B-Instruct-GGUF"
      filename: "Qwen2-Audio-7B-Instruct.Q4_K_M.gguf"
    mmproj:
      repo: "mradermacher/Qwen2-Audio-7B-Instruct-GGUF"
      filename: "Qwen2-Audio-7B-Instruct.mmproj-f16.gguf"
    system_prompt_type: "ChatML"
```

```python
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llava_node",
            name="llava_node",
            namespace="llama",
            parameters=[os.path.join(
                get_package_share_directory("llama_bringup"),
                "models", "Qwen2-Audio.yaml")],
        )
    ])
```

```shell
ros2 launch llama_bringup llava.launch.py
```

</details>

### ROS 2 Parameters

The following tables list all the ROS 2 parameters available when launching `llama_node` or `llava_node`. Parameters are organized by namespace.

#### General

| Param             | Type       | Default   | Description                                                                     |
| ----------------- | ---------- | --------- | ------------------------------------------------------------------------------- |
| `verbosity`       | `int32`    | `3`       | Log verbosity level                                                             |
| `seed`            | `int32`    | `-1`      | RNG seed for sampling (`-1` for default)                                        |
| `n_ctx`           | `int32`    | `0`       | Context size in tokens (`0` for model default)                                  |
| `n_batch`         | `int32`    | `2048`    | Logical batch size for prompt processing                                        |
| `n_ubatch`        | `int32`    | `512`     | Physical batch size                                                             |
| `n_keep`          | `int32`    | `0`       | Number of tokens to keep from the initial prompt on context shift               |
| `n_chunks`        | `int32`    | `-1`      | Max number of chunks to process (`-1` for unlimited)                            |
| `n_predict`       | `int32`    | `-1`      | Max tokens to predict (`-1` for unlimited)                                      |
| `n_parallel`      | `int32`    | `1`       | Number of parallel sequences to decode                                          |
| `n_sequences`     | `int32`    | `1`       | Number of sequences to decode                                                   |
| `n_gpu_layers`    | `int32`    | `-1`      | Number of layers to offload to GPU (`-1` for all)                               |
| `main_gpu`        | `int32`    | `0`       | Main GPU index                                                                  |
| `split_mode`      | `string`   | `"layer"` | GPU split mode: `none`, `layer`, or `row`                                       |
| `tensor_split`    | `double[]` | `[0.0]`   | Tensor split proportions across GPUs                                            |
| `devices`         | `string[]` | `[]`      | GPU device names to use                                                         |
| `numa`            | `string`   | `"none"`  | NUMA strategy: `none`, `distribute`, `isolate`, `numactl`, `mirror`, or `count` |
| `flash_attn_type` | `string`   | `"auto"`  | Flash attention type: `auto`, `enabled`, or `disabled`                          |
| `pooling_type`    | `string`   | `""`      | Pooling type: `none`, `mean`, `cls`, `last`, or `rerank`                        |
| `attention_type`  | `string`   | `""`      | Attention type: `causal` or `non_causal`                                        |
| `embedding`       | `bool`     | `false`   | Enable embedding mode                                                           |
| `reranking`       | `bool`     | `false`   | Enable reranking mode (sets pooling to `rerank` and enables embedding)          |
| `use_mmap`        | `bool`     | `true`    | Use memory-mapped files for model loading                                       |
| `use_direct_io`   | `bool`     | `false`   | Use direct I/O for model loading                                                |
| `use_mlock`       | `bool`     | `false`   | Lock model in RAM to prevent swapping                                           |
| `warmup`          | `bool`     | `true`    | Run a warmup inference on load                                                  |
| `check_tensors`   | `bool`     | `false`   | Validate model tensor data on load                                              |
| `ctx_shift`       | `bool`     | `false`   | Enable context shifting                                                         |
| `swa_full`        | `bool`     | `false`   | Enable full sliding window attention                                            |
| `no_op_offload`   | `bool`     | `false`   | Disable operation offloading                                                    |
| `no_extra_bufts`  | `bool`     | `false`   | Disable extra buffer types                                                      |
| `no_kv_offload`   | `bool`     | `false`   | Disable KV cache offloading to GPU                                              |
| `no_host`         | `bool`     | `false`   | Disable host buffer usage                                                       |
| `kv_unified`      | `bool`     | `false`   | Use unified KV cache                                                            |
| `cont_batching`   | `bool`     | `true`    | Enable continuous batching                                                      |

#### Model (`model.*`)

| Param            | Type     | Default | Description                                          |
| ---------------- | -------- | ------- | ---------------------------------------------------- |
| `model.path`     | `string` | `""`    | Local file path to the GGUF model                    |
| `model.repo`     | `string` | `""`    | HuggingFace repository ID to download the model from |
| `model.filename` | `string` | `""`    | Filename of the model in the HuggingFace repository  |

#### Multimodal Projector (`mmproj.*`)

| Param             | Type     | Default | Description                                              |
| ----------------- | -------- | ------- | -------------------------------------------------------- |
| `mmproj.path`     | `string` | `""`    | Local file path to the multimodal projector              |
| `mmproj.repo`     | `string` | `""`    | HuggingFace repository ID to download the projector from |
| `mmproj.filename` | `string` | `""`    | Filename of the projector in the HuggingFace repository  |
| `mmproj.use_gpu`  | `bool`   | `true`  | Use GPU for the multimodal projector                     |
| `mmproj.disabled` | `bool`   | `false` | Disable loading the multimodal projector                 |

#### CPU (`cpu.*`)

| Param           | Type     | Default    | Description                                                                  |
| --------------- | -------- | ---------- | ---------------------------------------------------------------------------- |
| `cpu.n_threads` | `int32`  | `-1`       | Number of threads for generation (`-1` for auto-detect)                      |
| `cpu.poll`      | `int32`  | `50`       | Thread pool polling interval                                                 |
| `cpu.mask`      | `string` | `""`       | CPU affinity mask for generation threads                                     |
| `cpu.range`     | `string` | `""`       | CPU range for generation threads                                             |
| `cpu.priority`  | `string` | `"normal"` | Thread scheduling priority: `low`, `normal`, `medium`, `high`, or `realtime` |
| `cpu.strict`    | `bool`   | `false`    | Strict CPU affinity for generation threads                                   |

#### CPU Batch (`cpu_batch.*`)

| Param                 | Type     | Default    | Description                                                   |
| --------------------- | -------- | ---------- | ------------------------------------------------------------- |
| `cpu_batch.n_threads` | `int32`  | `-1`       | Number of threads for batch processing (`-1` for auto-detect) |
| `cpu_batch.poll`      | `int32`  | `50`       | Thread pool polling interval for batch processing             |
| `cpu_batch.mask`      | `string` | `""`       | CPU affinity mask for batch processing threads                |
| `cpu_batch.range`     | `string` | `""`       | CPU range for batch processing threads                        |
| `cpu_batch.priority`  | `string` | `"normal"` | Thread scheduling priority for batch processing               |
| `cpu_batch.strict`    | `bool`   | `false`    | Strict CPU affinity for batch processing threads              |

#### RoPE (`rope.*`)

| Param               | Type     | Default | Description                                                |
| ------------------- | -------- | ------- | ---------------------------------------------------------- |
| `rope.freq_base`    | `float`  | `0.0`   | RoPE base frequency (`0.0` for model default)              |
| `rope.freq_scale`   | `float`  | `0.0`   | RoPE frequency scale factor (`0.0` for model default)      |
| `rope.scaling_type` | `string` | `""`    | RoPE scaling type: `none`, `linear`, `yarn`, or `longrope` |

#### YaRN (`yarn.*`)

| Param              | Type    | Default | Description                                              |
| ------------------ | ------- | ------- | -------------------------------------------------------- |
| `yarn.ext_factor`  | `float` | `-1.0`  | YaRN extrapolation mix factor (`-1.0` for model default) |
| `yarn.attn_factor` | `float` | `-1.0`  | YaRN attention magnitude scaling factor                  |
| `yarn.beta_fast`   | `float` | `-1.0`  | YaRN low correction dimension                            |
| `yarn.beta_slow`   | `float` | `-1.0`  | YaRN high correction dimension                           |
| `yarn.orig_ctx`    | `int32` | `0`     | YaRN original context size                               |

#### Group Attention (`grp_attn.*`)

| Param        | Type    | Default | Description                                           |
| ------------ | ------- | ------- | ----------------------------------------------------- |
| `grp_attn.n` | `int32` | `1`     | Self-extend group attention factor (`1` for disabled) |
| `grp_attn.w` | `int32` | `512`   | Self-extend group attention width                     |

#### KV Cache (`cache.*`)

| Param          | Type     | Default | Description                                                                                      |
| -------------- | -------- | ------- | ------------------------------------------------------------------------------------------------ |
| `cache.type_k` | `string` | `"f16"` | Data type for K cache: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, or `q5_1` |
| `cache.type_v` | `string` | `"f16"` | Data type for V cache (same options as `cache.type_k`)                                           |

#### Fit Parameters (`fit.*`)

| Param         | Type    | Default | Description                                            |
| ------------- | ------- | ------- | ------------------------------------------------------ |
| `fit.enabled` | `bool`  | `true`  | Automatically fit model parameters to available memory |
| `fit.min_ctx` | `int32` | `4096`  | Minimum context size when fitting parameters           |

#### Speculative Decoding (`speculative.*`)

Speculative decoding uses a smaller draft model to predict multiple tokens ahead, then verifies them in parallel with the main model. This can significantly speed up text generation, especially when using a large target model with a smaller draft model from the same model family.

**Note:** Speculative decoding requires `n_parallel: 1` (single slot) and is not supported with embedding/reranking models.

| Param                        | Type     | Default  | Description                                                                                                                                                                                             |
| ---------------------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `speculative.type`           | `string` | `"none"` | Speculative decoding type: `none`, `draft`, `eagle3`, `ngram_simple`, `ngram_map_k`, `ngram_map_k4v`, `ngram_mod`, or `ngram_cache`                                                                     |
| `speculative.n_max`          | `int32`  | `16`     | Maximum number of tokens to draft per speculative step                                                                                                                                                  |
| `speculative.n_min`          | `int32`  | `0`      | Minimum number of draft tokens required to attempt verification. If the draft model produces fewer tokens than this, the draft is discarded and a single token is generated instead. `0` is recommended |
| `speculative.p_min`          | `double` | `0.75`   | Minimum probability threshold for draft tokens (greedy)                                                                                                                                                 |
| `speculative.n_ctx`          | `int32`  | `0`      | Context size for the draft model (`0` for same as target)                                                                                                                                               |
| `speculative.n_gpu_layers`   | `int32`  | `-1`     | Number of layers to offload to GPU for the draft model (`-1` for all)                                                                                                                                   |
| `speculative.model.path`     | `string` | `""`     | Local file path to the draft model GGUF file                                                                                                                                                            |
| `speculative.model.repo`     | `string` | `""`     | HuggingFace repository ID for the draft model                                                                                                                                                           |
| `speculative.model.filename` | `string` | `""`     | Filename of the draft model in the HuggingFace repository                                                                                                                                               |

#### Prompt & Chat

| Param                | Type       | Default | Description                                                               |
| -------------------- | ---------- | ------- | ------------------------------------------------------------------------- |
| `prefix`             | `string`   | `""`    | Text prepended to every user prompt                                       |
| `suffix`             | `string`   | `""`    | Text appended to every user prompt                                        |
| `system_prompt`      | `string`   | `""`    | Initial system prompt                                                     |
| `system_prompt_file` | `string`   | `""`    | Path to a file containing the system prompt                               |
| `system_prompt_type` | `string`   | `""`    | System prompt type (loads from a predefined YAML in `llama_ros/prompts/`) |
| `chat_template_file` | `string`   | `""`    | Path to a Jinja chat template file                                        |
| `stopping_words`     | `string[]` | `[]`    | List of words/tokens that stop generation                                 |

#### LoRA Adapters

| Param                     | Type       | Default | Description                                         |
| ------------------------- | ---------- | ------- | --------------------------------------------------- |
| `loras`                   | `string[]` | `[]`    | List of LoRA adapter names to load                  |
| `lora_init_without_apply` | `bool`     | `false` | Load LoRA adapters without applying them            |
| `<lora_name>.repo`        | `string`   | `""`    | HuggingFace repository for the LoRA adapter         |
| `<lora_name>.filename`    | `string`   | `""`    | Filename of the LoRA adapter in the repository      |
| `<lora_name>.file_path`   | `string`   | `""`    | Local file path to the LoRA adapter                 |
| `<lora_name>.scale`       | `double`   | `1.0`   | LoRA adapter scale factor (clamped to `[0.0, 1.0]`) |

### LoRA Adapters

You can use LoRA adapters when launching LLMs. Using llama.cpp features, you can load multiple adapters choosing the scale to apply for each adapter. Here you have an example of using LoRA adapters with Phi-3. You can list the
LoRAs using the `/llama/list_loras` service and modify their scales values by using the `/llama/update_loras` service. A scale value of 0.0 means not using that LoRA.

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 2048
    n_batch: 8
    n_predict: 2048
    n_gpu_layers: 0
    cpu:
      n_threads: 1
    model:
      repo: bartowski/Phi-3.5-mini-instruct-GGUF
      filename: Phi-3.5-mini-instruct-Q4_K_M.gguf
    loras:
      - code_writer
      - summarization
    code_writer:
      repo: zhhan/adapter-Phi-3-mini-4k-instruct_code_writing
      filename: Phi-3-mini-4k-instruct-adaptor-f16-code_writer.gguf
      scale: 0.5
    summarization:
      repo: zhhan/adapter-Phi-3-mini-4k-instruct_summarization
      filename: Phi-3-mini-4k-instruct-adaptor-f16-summarization.gguf
      scale: 0.5
    system_prompt_type: Phi-3
```

</details>

### ROS 2 Clients

Both llama_ros and llava_ros provide ROS 2 interfaces to access the main functionalities of the models. Here you have some examples of how to use them inside ROS 2 nodes. Moreover, take a look to the [llama_demo_node.py](llama_demos/llama_demos/llama_demo_node.py) and [llava_demo_node.py](llama_demos/llama_demos/llava_demo_node.py) demos.

#### Tokenize

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import Tokenize


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(Tokenize, "/llama/tokenize")

        # create the request
        req = Tokenize.Request()
        req.text = "Example text"

        # call the tokenize service
        self.srv_client.wait_for_service()
        tokens = self.srv_client.call(req).tokens
```

</details>

#### Detokenize

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import Detokenize


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(Detokenize, "/llama/detokenize")

        # create the request
        req = Detokenize.Request()
        req.tokens = [123, 123]

        # call the tokenize service
        self.srv_client.wait_for_service()
        text = self.srv_client.call(req).text
```

</details>

#### Embeddings

<details>
<summary>Click to expand</summary>

_Remember to launch llama_ros with embedding set to true to be able of generating embeddings with your LLM._

```python
from rclpy.node import Node
from llama_msgs.srv import GenerateEmbeddings


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(GenerateEmbeddings, "/llama/generate_embeddings")

        # create the request
        req = GenerateEmbeddings.Request()
        req.prompt = "Example text"
        req.normalization = 2  # -1=none, 0=max abs int16, 1=taxicab, 2=euclidean, >2=p-norm

        # call the embedding service
        self.srv_client.wait_for_service()
        embeddings = self.srv_client.call(req).embeddings
```

</details>

#### Generate Response

<details>
<summary>Click to expand</summary>

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateResponse


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.action_client = ActionClient(
            self, GenerateResponse, "/llama/generate_response")

        # create the goal and set the sampling config
        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(
            goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result: GenerateResponse.Result = get_result_future.result().result
```

</details>

#### Generate Response (llava)

<details>
<summary>Click to expand</summary>

```python
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateResponse


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create a cv bridge for the image
        self.cv_bridge = CvBridge()

        # create the client
        self.action_client = ActionClient(
            self, GenerateResponse, "/llama/generate_response")

        # create the goal and set the sampling config
        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2

        # add your image to the goal
        image = cv2.imread("/path/to/your/image", cv2.IMREAD_COLOR)
        goal.images.append(self.cv_bridge.cv2_to_imgmsg(image))

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(
            goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result: GenerateResponse.Result = get_result_future.result().result
```

</details>

#### Generate Chat Completions

<details>
<summary>Click to expand</summary>

The `GenerateChatCompletions` action provides an OpenAI-compatible chat completions interface with support for tool calling, reasoning, and streaming.

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GenerateChatCompletions
from llama_msgs.msg import ChatMessage


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.action_client = ActionClient(
            self, GenerateChatCompletions, "/llama/generate_chat_completions")

        # create the goal
        goal = GenerateChatCompletions.Goal()
        goal.messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is ROS 2?")
        ]
        goal.sampling_config.temp = 0.2
        goal.stream = True

        # wait for the server and send the goal
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal)

        # wait for the server
        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        # wait again and take the result
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
```

</details>

#### Get Metadata

<details>
<summary>Click to expand</summary>

```python
from rclpy.node import Node
from llama_msgs.srv import GetMetadata


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(GetMetadata, "/llama/get_metadata")

        # call the metadata service
        req = GetMetadata.Request()
        self.srv_client.wait_for_service()
        metadata = self.srv_client.call(req).metadata
```

</details>

#### Rerank Documents

<details>
<summary>Click to expand</summary>

_Remember to launch llama_ros with reranking set to true._

```python
from rclpy.node import Node
from llama_msgs.srv import RerankDocuments


class ExampleNode(Node):
    def __init__(self) -> None:
        super().__init__("example_node")

        # create the client
        self.srv_client = self.create_client(RerankDocuments, "/llama/rerank_documents")

        # create the request
        req = RerankDocuments.Request()
        req.query = "What is robotics?"
        req.documents = ["Robotics is a field of engineering.", "The weather is sunny."]

        # call the reranking service
        self.srv_client.wait_for_service()
        scores = self.srv_client.call(req).scores
```

</details>

### LangChain

There is a [llama_ros integration for LangChain](llama_ros/llama_ros/langchain/). Thus, prompt engineering techniques could be applied. Here you have an example to use it.

#### llama_ros (Chain)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# create a prompt template
prompt_template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template
)

# create a chain with the llm and the prompt template
chain = prompt | llm | StrOutputParser()

# run the chain
text = chain.invoke({"topic": "bears"})
print(text)

rclpy.shutdown()
```

</details>

#### llama_ros (Stream)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# create a prompt template
prompt_template = "tell me a joke about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template
)

# create a chain with the llm and the prompt template
chain = prompt | llm | StrOutputParser()

# run the chain
for c in chain.stream({"topic": "bears"}):
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

#### llava_ros

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROS

rclpy.init()

# create the llama_ros llm for langchain
llm = LlamaROS()

# bind the url_image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
llm = llm.bind(image_url=image_url).stream("Describe the image")

# run the llm
for c in llm:
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

#### llama_ros_embeddings (RAG)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from langchain_chroma import Chroma
from llama_ros.langchain import LlamaROSEmbeddings


rclpy.init()

# create the llama_ros embeddings for langchain
embeddings = LlamaROSEmbeddings()

# create a vector database and assign it
db = Chroma(embedding_function=embeddings)

# create the retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# add your texts
db.add_texts(texts=["your_texts"])

# retrieve documents
documents = retriever.invoke("your_query")
print(documents)

rclpy.shutdown()
```

</details>

#### llama_ros (Reranker)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import LlamaROSReranker
from llama_ros.langchain import LlamaROSEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever


rclpy.init()

# load the documents
documents = TextLoader("../state_of_the_union.txt",).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# create the llama_ros embeddings
embeddings = LlamaROSEmbeddings()

# create the VD and the retriever
retriever = FAISS.from_documents(
    texts, embeddings).as_retriever(search_kwargs={"k": 20})

# create the compressor using the llama_ros reranker
compressor = LlamaROSReranker()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# retrieve the documents
compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)

for doc in compressed_docs:
    print("-" * 50)
    print(doc.page_content)
    print("\n")

rclpy.shutdown()
```

</details>

#### llama_ros (LLM + RAG + Reranker)

<details>
<summary>Click to expand</summary>

```python
import bs4
import rclpy

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from llama_ros.langchain import ChatLlamaROS, LlamaROSEmbeddings, LlamaROSReranker


rclpy.init()

# load, chunk and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=LlamaROSEmbeddings())

# retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are an AI assistant that answer questions briefly."),
        HumanMessagePromptTemplate.from_template(
            "Taking into account the followin information:{context}\n\n{question}"
        ),
    ]
)

# create rerank compression retriever
compressor = LlamaROSReranker(top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


def format_docs(docs):
    formated_docs = ""

    for d in docs:
        formated_docs += f"\n\n\t- {d.page_content}"

    return formated_docs


# create and use the chain
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatLlamaROS(temp=0.0)
    | StrOutputParser()
)

for c in rag_chain.stream("What is Task Decomposition?"):
    print(c, flush=True, end="")

rclpy.shutdown()
```

</details>

#### chat_llama_ros (Chat + VLM)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


rclpy.init()

# create chat
chat = ChatLlamaROS(
    temp=0.2,
    penalty_last_n=8
)

# create prompt template with messages
prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are a IA that just answer with a single word."),
    HumanMessagePromptTemplate.from_template(template=[
        {"type": "text", "text": "<__media__>Who is the character in the middle of the image?"},
        {"type": "image_url", "image_url": "{image_url}"}
    ])
])

# create the chain
chain = prompt | chat | StrOutputParser()

# stream and print the LLM output
for text in chain.stream({"image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg"}):
    print(text, end="", flush=True)

print("", end="\n", flush=True)

rclpy.shutdown()
```

</details>

#### chat_llama_ros (Chat + Audio)

<details>
<summary>Click to expand</summary>

```python
import sys
import time
import rclpy
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_ros.langchain import ChatLlamaROS


def main():
    if len(sys.argv) < 2:
        prompt = "What's that sound?"
    else:
        prompt = " ".join(sys.argv[1:])

    tokens = 0
    initial_time = -1
    eval_time = -1

    rclpy.init()
    chat = ChatLlamaROS(temp=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are an IA that answer questions."),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": f"<__media__>{prompt}"},
                    {"type": "image_url", "image_url": "{audio_url}"},
                ]
            ),
        ]
    )

    chain = prompt | chat | StrOutputParser()

    initial_time = time.time()
    for text in chain.stream(
        {
            "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        }
    ):
        tokens += 1
        print(text, end="", flush=True)
        if eval_time < 0:
            eval_time = time.time()

    print("", end="\n", flush=True)

    end_time = time.time()
    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

</details>

#### chat_llama_ros (Structured output)

<details>
<summary>Click to expand</summary>

```python
import rclpy
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from llama_ros.langchain import ChatLlamaROS
from pydantic import BaseModel, Field

rclpy.init()

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

structured_chat = chat.with_structured_output(
    Joke, method="function_calling"
)

prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": "{prompt}"},
            ]
        ),
    ]
)

chain = prompt | structured_chat

res = chain.invoke({"prompt": "Tell me a joke about cats"})

print(f"Response: {res}")

rclpy.shutdown()
```

</details>

#### chat_llama_ros (Tools)

<details>
<summary>Click to expand</summary>

The current implementation of Tools allows executing tools without requiring a model trained for that task.

```python
from random import randint

import rclpy

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llama_ros.langchain import ChatLlamaROS

rclpy.init()

@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

messages = [
    HumanMessage(
        "What is the current temperature in Madrid? And its inhabitants?"
    )
]

llm_tools = chat.bind_tools(
    [get_inhabitants, get_curr_temperature], tool_choice='any'
)

all_tools_res = llm_tools.invoke(messages)
messages.append(all_tools_res)

for tool in all_tools_res.tool_calls:
    selected_tool = {
        "get_inhabitants": get_inhabitants, "get_curr_temperature": get_curr_temperature
    }[tool['name']]

    tool_msg = selected_tool.invoke(tool)

    formatted_output = f"{tool['name']}({''.join(tool['args'].values())}) = {tool_msg.content}"

    tool_msg.additional_kwargs = {'args': tool['args']}
    messages.append(tool_msg)

res = llm_tools.invoke(messages)

print(f"Response: {res.content}")

rclpy.shutdown()
```

</details>

#### chat_llama_ros (Reasoning)

<details>
<summary>Click to expand</summary>

A reasoning model is required, such as Deepseek R1

```python
import time
from random import randint

import rclpy

from langchain_core.messages import HumanMessage
from llama_ros.langchain import ChatLlamaROS

rclpy.init()

chat = ChatLlamaROS(temp=0.6, penalty_last_n=8)

messages = [
    HumanMessage(
        "Here we have a book, a laptop, 9 eggs and a nail. Please tell me how to stack them onto each other in a stable manner."
    )
]

res = chat.invoke(messages)

print(f"Response: {res.content.strip()}")
print(f"Reasoning: {res.additional_kwargs["reasoning_content"]}")

rclpy.shutdown()
```

</details>

#### chat_llama_ros (LangGraph)

<details>
<summary>Click to expand</summary>

```python
import time
from random import randint

import rclpy

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_ros.langchain import ChatLlamaROS

rclpy.init()

@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)

chat = ChatLlamaROS(temp=0.0)

agent_executor = create_react_agent(
    chat, [get_inhabitants, get_curr_temperature]
)

response = agent_executor.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is the current temperature in Madrid? And its inhabitants?"
            )
        ]
    }
)

print(f"Response: {response['messages'][-1].content}")

rclpy.shutdown()
```

</details>

## Demos

### LLM Demo

```shell
ros2 launch llama_bringup spaetzle.launch.py
```

```shell
ros2 run llama_demos llama_demo_node
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->

https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

### Speculative Decoding Demo

Launch the speculative decoding model using a Llama 3.1 8B target model with a Llama 3.2 1B draft model:

```shell
ros2 launch llama_bringup llama-3-speculative.launch.py
```

Then run any of the text generation demos, for example:

```shell
ros2 run llama_demos llama_demo_node
```

### Embeddings Generation Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 run llama_demos llama_embeddings_demo_node
```

https://github.com/user-attachments/assets/7d722017-27dc-417c-ace7-bf6b747e4ced

### Reranking Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 run llama_demos llama_rerank_demo_node
```

https://github.com/user-attachments/assets/4b4adb4d-7c70-43ea-a2c1-9be57d211484

### VLM Demo

```shell
ros2 launch llama_bringup minicpm-2.6.launch.py
```

```shell
ros2 run llama_demos llava_demo_node
```

https://github.com/mgonzs13/llama_ros/assets/25979134/4a9ef92f-9099-41b4-8350-765336e3503c

### Chat Template Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: 20
    cpu:
      n_threads: -1
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
```

</details>

```shell
ros2 run llama_demos chatllama_demo_node
```

[ChatLlamaROS demo](https://github-production-user-asset-6210df.s3.amazonaws.com/55236157/363094669-c6de124a-4e91-4479-99b6-685fecb0ac20.webm?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240830%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240830T081232Z&X-Amz-Expires=300&X-Amz-Signature=f937758f4bcbaec7683e46ddb057fb642dc86a33cc8c736fca3b5ce2bf06ddac&X-Amz-SignedHeaders=host&actor_id=55236157&key_id=0&repo_id=622137360)

### Chat Structured Output Demo

```shell
ros2 llama launch Qwen2.yaml
```

<details>
<summary>Click to expand Qwen2.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 2048
    n_batch: 8
    n_predict: 2048
    n_gpu_layers: -1
    cpu:
      n_threads: 1
    model:
      repo: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
      filename: qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_structured_demo_node
```

[Structured Output ChatLlama](https://github.com/user-attachments/assets/e0bf4031-50c0-4790-94a0-1f6aed5734ec)

### Chat Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_tools_demo_node
```

[Tools ChatLlama](https://github.com/user-attachments/assets/b912ee29-1466-4d6a-888b-9a2d9c16ae1d)

### Chat Reasoning Demo (DeepSeek-R1)

```shell
ros2 llama launch DeepSeek-R1.yaml
```

<details>
<summary>Click to expand DeepSeek-R1.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: 1
    model:
      repo: unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF
      filename: DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    chat_template_file: llama-cpp-deepseek-r1.jinja
```

</details>

```shell
ros2 run llama_demos chatllama_reasoning_demo_node
```

[DeepSeekR1 ChatLlama](https://github.com/user-attachments/assets/3f268614-eabc-4499-b50f-a76d76908d9d)

### Streaming Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_streaming_tools_demo_node
```

### Reasoning + Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_reasoning_tools_demo_node
```

### Multi-Image Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: 20
    cpu:
      n_threads: -1
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
```

</details>

```shell
ros2 run llama_demos chatllama_multi_image_demo_node
```

### Multi-Image (User Input) Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: 20
    cpu:
      n_threads: -1
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
```

</details>

```shell
ros2 run llama_demos chatllama_multi_image_user_demo_node
```

### Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
    system_prompt_type: ChatML
```

</details>

```shell
ros2 run llama_demos chatllama_audio_demo_node
```

### Multi-Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
```

</details>

```shell
ros2 run llama_demos chatllama_multi_audio_demo_node
```

### MTMD Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 8192
    n_batch: 512
    n_predict: 8192
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
    system_prompt_type: ChatML
```

</details>

```shell
ros2 run llama_demos mtmd_audio_demo_node
```

### PDDL Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_pddl_demo_node
```

### LangGraph Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: -1
    cpu:
      n_threads: -1
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_langgraph_demo_node
```

[Langgraph ChatLlama](https://github.com/user-attachments/assets/a0991cb4-f7f4-43d5-b629-3b1819aead0d)

### RAG Demo (LLM + chat template + RAG + Reranking + Stream)

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 llama launch Qwen2.yaml
```

<details>
<summary>Click to expand Qwen2.yaml</summary>

```yaml
/**:
  ros__parameters:
    n_ctx: 4096
    n_batch: 256
    n_predict: -1
    n_gpu_layers: 29
    cpu:
      n_threads: -1
    model:
      repo: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
      filename: "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos llama_rag_demo_node
```

https://github.com/user-attachments/assets/b4e3957d-1f92-427b-a1a8-cfc76737c0d6
