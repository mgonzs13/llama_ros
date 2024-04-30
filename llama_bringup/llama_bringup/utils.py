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
from typing import List
from ament_index_python.packages import get_package_share_directory
from huggingface_hub import hf_hub_download
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def download_model(repo: str, file: str) -> str:

    if repo and file:
        return hf_hub_download(repo_id=repo, filename=file, force_download=False)

    return ""


def get_prompt_path(prompt_file_name: str) -> str:

    if prompt_file_name:
        return os.path.join(
            get_package_share_directory("llama_bringup"),
            "prompts",
            f"{prompt_file_name}.txt"
        )

    return ""


def create_llama_launch(
    use_llava: bool = False,

    seed: int = -1,
    n_ctx: int = 512,
    n_batch: int = 2048,
    n_ubatch: int = 512,

    n_gpu_layers: int = 0,
    split_mode: str = "layer",
    main_gpu: int = 0,
    tensor_split: str = "[0.0]",

    grp_attn_n: int = 1,
    grp_attn_w: int = 512,

    rope_freq_base: float = 0.0,
    rope_freq_scale: float = 0.0,
    rope_scaling_type: str = "",

    yarn_ext_factor: float = -1.0,
    yarn_attn_factor: float = 1.0,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
    yarn_orig_ctx: float = 0,

    embedding: bool = True,
    logits_all: bool = False,
    use_mmap: bool = True,
    use_mlock: bool = False,
    warmup: bool = True,
    check_tensors: bool = False,
    flash_attn: bool = False,

    dump_kv_cache: bool = False,
    no_kv_offload: bool = False,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",

    n_threads: int = 4,
    n_threads_batch: int = -1,
    n_predict: int = 128,
    n_keep: int = -1,

    model: str = "",
    model_repo: str = "",
    model_filename: str = "",

    lora_base: str = "",
    lora_base_repo: str = "",
    lora_base_filename: str = "",

    mmproj: str = "",
    mmproj_repo: str = "",
    mmproj_filename: str = "",

    numa: str = "none",
    pooling_type: str = "",

    prefix: str = "",
    suffix: str = "",
    stopping_words: List[str] = [],

    system_prompt: str = "",
    system_prompt_file: str = "",
    system_prompt_type: str = "",
    debug: bool = True
) -> IncludeLaunchDescription:

    if not system_prompt_file and system_prompt_type:
        system_prompt_file = get_prompt_path(system_prompt_type)

    if not model:
        model = download_model(model_repo, model_filename)

    if not lora_base:
        lora_base = download_model(lora_base_repo, lora_base_filename)

    if not mmproj:
        mmproj = download_model(mmproj_repo, mmproj_filename)

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("llama_bringup"),
                "launch",
                "base.launch.py"
            )
        ),
        launch_arguments={
            "use_llava": str(use_llava),

            "seed": str(seed),
            "n_ctx": str(n_ctx),
            "n_batch": str(n_batch),
            "n_ubatch": str(n_ubatch),

            "n_gpu_layers": str(n_gpu_layers),
            "split_mode": split_mode,
            "main_gpu": str(main_gpu),
            "tensor_split": tensor_split,

            "grp_attn_n": str(grp_attn_n),
            "grp_attn_w": str(grp_attn_w),

            "rope_freq_base": str(rope_freq_base),
            "rope_freq_scale": str(rope_freq_scale),
            "yarn_ext_factor": str(yarn_ext_factor),
            "yarn_attn_factor": str(yarn_attn_factor),
            "yarn_beta_fast": str(yarn_beta_fast),
            "yarn_beta_slow": str(yarn_beta_slow),
            "yarn_orig_ctx": str(yarn_orig_ctx),
            "rope_scaling_type": str(rope_scaling_type),

            "embedding": str(embedding),
            "logits_all": str(logits_all),
            "use_mmap": str(use_mmap),
            "use_mlock": str(use_mlock),
            "warmup": str(warmup),
            "check_tensors": str(check_tensors),
            "flash_attn": str(flash_attn),

            "dump_kv_cache": str(dump_kv_cache),
            "no_kv_offload": str(no_kv_offload),
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,

            "n_threads": str(n_threads),
            "n_threads_batch": str(n_threads_batch),
            "n_predict": str(n_predict),
            "n_keep": str(n_keep),

            "model": model,
            "lora_base": lora_base,
            "mmproj": mmproj,
            "numa": numa,
            "pooling_type": pooling_type,

            "prefix": prefix,
            "suffix": suffix,
            "stopping_words": str(stopping_words),

            "system_prompt": system_prompt,
            "system_prompt_file": system_prompt_file,
            "debug": str(debug)
        }.items()
    )
