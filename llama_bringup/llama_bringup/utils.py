# MIT License
#
# Copyright (c) 2023 Miguel Ángel González Santamarta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import re
import yaml
from typing import Tuple
from huggingface_hub import hf_hub_download
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def download_model(repo: str, file: str) -> str:

    match = re.search(r"-(\d+)-of-(\d+)\.gguf", file)

    if match:
        total_shards = int(match.group(2))
        base_name = file[: match.start()]

        # download shards
        for i in range(1, total_shards + 1):
            shard_file = f"{base_name}-{i:05d}-of-{total_shards:05d}.gguf"
            hf_hub_download(repo_id=repo, filename=shard_file, force_download=False)

        # return first shard
        return hf_hub_download(
            repo_id=repo,
            filename=f"{base_name}-00001-of-{total_shards:05d}.gguf",
            force_download=False,
        )

    return hf_hub_download(repo_id=repo, filename=file, force_download=False)


def load_prompt_type(prompt_file_name: str) -> Tuple:
    file_path = os.path.join(
        get_package_share_directory("llama_bringup"),
        "prompts",
        f"{prompt_file_name}.yaml",
    )
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    return (
        yaml_data["prefix"],
        yaml_data["suffix"],
        yaml_data["stopping_words"],
        yaml_data["system_prompt"],
    )


def create_llama_launch_from_yaml(file_path: str) -> IncludeLaunchDescription:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return create_llama_launch(**config)


def create_llama_launch(**kwargs) -> IncludeLaunchDescription:
    prompt_data = (
        load_prompt_type(kwargs["system_prompt_type"])
        if kwargs.get("system_prompt_type")
        else ("", "", [], "")
    )
    kwargs["prefix"] = kwargs.get("prefix", prompt_data[0])
    kwargs["suffix"] = kwargs.get("suffix", prompt_data[1])
    kwargs["system_prompt"] = kwargs.get("system_prompt", prompt_data[3])

    # stopping_words
    kwargs["stopping_words"] = kwargs.get("stopping_words", prompt_data[2])
    if not kwargs["stopping_words"]:
        kwargs["stopping_words"] = [""]

    # load lora adapters
    lora_adapters = []
    lora_adapters_scales = []

    if "lora_adapters" in kwargs:
        for i in range(len(kwargs["lora_adapters"])):
            lora = kwargs["lora_adapters"][i]

            if "repo" in lora and "filename" in lora:
                lora_path = download_model(lora["repo"], lora["filename"])

            elif "path" in lora:
                lora_path = lora["path"]

            else:
                continue

            if "scale" not in lora:
                continue

            lora_adapters.append(lora_path)
            lora_adapters_scales.append(lora["scale"])

    else:
        lora_adapters = [""]
        lora_adapters_scales = [0.0]

    kwargs["lora_adapters"] = lora_adapters
    kwargs["lora_adapters_scales"] = lora_adapters_scales

    # use llava
    if not kwargs.get("use_llava"):
        kwargs["use_llava"] = False

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("llama_bringup"), "launch", "base.launch.py"
            )
        ),
        launch_arguments={key: str(value) for key, value in kwargs.items()}.items(),
    )
