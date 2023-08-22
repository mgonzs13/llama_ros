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

    if prompt_file_name:

        return os.path.join(
            get_package_share_directory("llama_bringup"),
            "prompts",
            prompt_file_name
        )

    return ""


def create_llama_launch(
    seed: int = -1,
    n_ctx: int = 512,
    n_batch: int = 8,

    n_gpu_layers: int = 0,
    main_gpu: int = 0,
    tensor_split: str = "[0.0]",

    rope_freq_base: float = 10000.0,
    rope_freq_scale: float = 1.0,

    low_vram: bool = False,
    mul_mat_q: bool = False,
    f16_kv: bool = True,
    logits_all: bool = False,
    vocab_only: bool = False,
    use_mmap: bool = True,
    use_mlock: bool = False,
    embedding: bool = True,

    n_threads: int = 4,
    n_predict: int = 128,
    n_keep: int = -1,

    model: str = "",
    lora_adapter: str = "",
    lora_base: str = "",
    numa: bool = True,

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
            "n_batch": str(n_batch),

            "n_gpu_layers": str(n_gpu_layers),
            "main_gpu": str(main_gpu),
            "tensor_split": tensor_split,

            "rope_freq_base": str(rope_freq_base),
            "rope_freq_scale": str(rope_freq_scale),

            "low_vram": str(low_vram),
            "mul_mat_q": str(mul_mat_q),
            "f16_kv": str(f16_kv),
            "logits_all": str(logits_all),
            "vocab_only": str(vocab_only),
            "use_mmap": str(use_mmap),
            "use_mlock": str(use_mlock),
            "embedding": str(embedding),

            "n_threads": str(n_threads),
            "n_predict": str(n_predict),
            "n_keep": str(n_keep),

            "model": get_llama_model_path(model),
            "lora_adapter": get_lora_model_path(lora_adapter),
            "lora_base": get_llama_model_path(lora_base),
            "numa": str(numa),

            "prefix": prefix,
            "suffix": suffix,
            "stop": stop,

            "prompt": prompt,
            "file": get_prompt_path(file),
        }.items()
    )
