# MIT License

# Copyright (c) 2023  Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def get_base_launch_path() -> str:
    return os.path.join(
        get_package_share_directory("llama_bringup"),
        "launch",
        "base.launch.py")


def get_llama_model_path(model_name: str) -> str:
    return os.path.join(
        os.path.abspath(os.path.normpath(
            os.path.expanduser("~/llama_models"))),
        model_name
    )


def get_lora_model_path(model_name: str) -> str:

    if model_name:

        return os.path.join(
            os.path.abspath(os.path.normpath(
                os.path.expanduser("~/llama_models"))),
            "lora",
            model_name
        )

    return ""


def get_prompt_path(prompt_file_name: str) -> str:
    return os.path.join(
        get_package_share_directory("llama_bringup"),
        "prompts",
        prompt_file_name
    )


def create_llama_launch(
    seed: int = -1,
    n_ctx: int = 512,
    memory_f16: bool = True,
    use_mmap: bool = True,
    use_mlock: bool = False,
    embedding: bool = True,

    n_gpu_layers: int = 0,
    main_gpu: int = 0,
    tensor_split: str = "[0.0]",

    n_threads: int = 4,
    n_predict: int = 128,
    n_batch: int = 8,
    n_keep: int = -1,

    temp: float = 0.80,
    top_k: int = 40,
    top_p: float = 0.95,
    tfs_z: float = 1.00,
    typical_p: float = 1.00,
    repeat_last_n: int = 64,
    repeat_penalty: float = 1.10,
    presence_penalty: float = 0.00,
    frequency_penalty: float = 0.00,
    mirostat: int = 0,
    mirostat_tau: float = 5.00,
    mirostat_eta: float = 0.10,
    penalize_nl: bool = True,

    model: str = "",
    lora_adapter: str = "",
    lora_base: str = "",

    prefix: str = "",
    suffix: str = "",
    stop: str = "",

    prompt: str = "",
    file: str = ""
) -> IncludeLaunchDescription:

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(get_base_launch_path()),
        launch_arguments={
            "seed": str(seed),
            "n_ctx": str(n_ctx),
            "memory_f16": str(memory_f16),
            "use_mmap": str(use_mmap),
            "use_mlock": str(use_mlock),
            "embedding": str(embedding),

            "n_gpu_layers": str(n_gpu_layers),
            "main_gpu": str(main_gpu),
            "tensor_split": tensor_split,

            "n_threads": str(n_threads),
            "n_predict": str(n_predict),
            "n_batch": str(n_batch),
            "n_keep": str(n_keep),

            "temp": str(temp),
            "top_k": str(top_k),
            "top_p": str(top_p),
            "tfs_z": str(tfs_z),
            "typical_p": str(typical_p),
            "repeat_last_n": str(repeat_last_n),
            "repeat_penalty": str(repeat_penalty),
            "presence_penalty": str(presence_penalty),
            "frequency_penalty": str(frequency_penalty),
            "mirostat": str(mirostat),
            "mirostat_tau": str(mirostat_tau),
            "mirostat_eta": str(mirostat_eta),
            "penalize_nl": str(penalize_nl),

            "model": get_llama_model_path(model),
            "lora_adapter": get_lora_model_path(lora_adapter),
            "lora_base": get_llama_model_path(lora_base),

            "prefix": prefix,
            "suffix": suffix,
            "stop": stop,

            "prompt": prompt,
            "file": get_prompt_path(file),
        }.items()
    )
