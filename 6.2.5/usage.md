# Usage

[← Back to README](../README.md)

## llama_cli

Commands are included in llama_ros to speed up the test of GGUF-based LLMs within the ROS 2 ecosystem. This way, the following commands are integrating into the ROS 2 commands:

### launch

Using this command launch a LLM from a YAML file. The configuration of the YAML is used to launch the LLM in the same way as using a regular launch file. Here is an example of how to use it:

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/StableLM-Zephyr.yaml
```

### prompt

Using this command send a prompt to a launched LLM. The command uses a string, which is the prompt and has the following arguments:

- (`-r`, `--reset`): Whether to reset the LLM before prompting
- (`-t`, `--temp`): The temperature value
- (`--image-url`): Image url to sent to a VLM

Here is an example of how to use it:

```shell
ros2 llama prompt "Do you know ROS 2?" -t 0.0
```

## Launch Files

First of all, you need to create a launch file to use llama_ros or llava_ros. This launch file will contain the main parameters to download the model from HuggingFace and configure it. Take a look at the following examples and the [predefined launch files](../llama_bringup/launch).

`llama_bringup` also ships a generic `base.launch.py` that takes `params_file` (a model YAML), `executable` (`llama_node` or `llava_node`), `node_name` and `namespace` arguments, so any predefined model YAML can be launched with it, e.g.:

```shell
ros2 launch llama_bringup base.launch.py params_file:=$(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/Llama-3.yaml executable:=llama_node
```

### llama_ros (Python Launch)

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
                "context.n_ctx": 2048,
                "context.n_batch": 8,
                "context.n_predict": 2048,
                "gpu.n_gpu_layers": 0,
                "cpu.n_threads": 1,
                "model.repo": "TheBloke/Marcoroni-7B-v3-GGUF",
                "model.filename": "marcoroni-7b-v3.Q4_K_M.gguf",
                "prompt.system_prompt_type": "Alpaca",
            }],
        )
    ])
```

```shell
ros2 launch <path/to/marcoroni.launch.py>
```

</details>

### llama_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "cstr/Spaetzle-v60-7b-GGUF"
      filename: "Spaetzle-v60-7b-q4-k-m.gguf"
    context:
      n_ctx: 2048
      n_batch: 8
      n_ubatch: 8
      n_predict: 2048
    gpu:
      n_gpu_layers: 0
    cpu:
      n_threads: 1
    prompt:
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

### llama_ros (YAML Config + model shards)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" # Hugging Face repo
      filename: "qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf" # model shard file in repo

    context:
      n_ctx: 2048 # context of the LLM in tokens
      n_batch: 8 # batch size in tokens
      n_ubatch: 8
      n_predict: 2048 # max tokens, -1 == inf

    gpu:
      n_gpu_layers: 0 # layers to load in GPU

    cpu:
      n_threads: 1 # threads

    prompt:
      system_prompt_type: "ChatML" # system prompt type
```

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/Qwen2.yaml
```

</details>

### llama_ros (Speculative Decoding)

<details>
<summary>Click to expand</summary>

[Speculative decoding](https://arxiv.org/abs/2302.01318) accelerates text generation by drafting candidate tokens and verifying them in parallel with the main model. llama_ros supports draft-model-based methods (`draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`) and self-speculative ngram-based methods (`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`) that require no separate model. Note that speculative decoding requires `context.n_parallel: 1`.

```yaml
/**:
  ros__parameters:
    model:
      repo: lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
      filename: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 2048
      n_predict: 2048
      n_parallel: 1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      system_prompt_type: Llama-3
    speculative:
      type: draft-simple
      draft:
        n_max: 16
        n_min: 0
        p_min: 0.75
        n_gpu_layers: -1
        model:
          repo: lmstudio-community/Llama-3.2-1B-Instruct-GGUF
          filename: Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

```shell
ros2 launch llama_bringup llama-3-speculative.launch.py
```

</details>

### llava_ros (Python Launch)

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
                "context.n_ctx": 8192,
                "context.n_batch": 512,
                "gpu.n_gpu_layers": 33,
                "cpu.n_threads": 1,
                "context.n_predict": 8192,
                "model.repo": "cjpais/llava-1.6-mistral-7b-gguf",
                "model.filename": "llava-v1.6-mistral-7b.Q4_K_M.gguf",
                "mmproj.repo": "cjpais/llava-1.6-mistral-7b-gguf",
                "mmproj.filename": "mmproj-model-f16.gguf",
                "prompt.system_prompt_type": "Mistral",
            }],
        )
    ])
```

```shell
ros2 launch <path/to/llava.launch.py>
```

</details>

### llava_ros (YAML Config)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "cjpais/llava-1.6-mistral-7b-gguf"
      filename: "llava-v1.6-mistral-7b.Q4_K_M.gguf"
    mmproj:
      repo: "cjpais/llava-1.6-mistral-7b-gguf"
      filename: "mmproj-model-f16.gguf"
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: 33
    cpu:
      n_threads: 1
    prompt:
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
ros2 launch llama_bringup base.launch.py params_file:=$(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/llava-mistral.yaml executable:=llava_node node_name:=llava_node
```

</details>

### llava_ros (Audio)

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "mradermacher/Qwen2-Audio-7B-Instruct-GGUF"
      filename: "Qwen2-Audio-7B-Instruct.Q4_K_M.gguf"
    mmproj:
      repo: "mradermacher/Qwen2-Audio-7B-Instruct-GGUF"
      filename: "Qwen2-Audio-7B-Instruct.mmproj-f16.gguf"
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: 29
    cpu:
      n_threads: -1
    prompt:
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
ros2 launch llama_bringup base.launch.py params_file:=$(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/Qwen2-Audio.yaml executable:=llava_node node_name:=llava_node
```

</details>
