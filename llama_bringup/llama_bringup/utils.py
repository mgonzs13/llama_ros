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
import yaml
from typing import Dict, Any, Tuple
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def load_params_from_yaml(file_path: str) -> Dict[str, Any]:
    """Read a flat YAML file and return its contents as a dict.

    Expected format:
        key: value
        key2: value2
    """
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return dict(yaml_data)


def _resolve_lora_adapters(params: Dict[str, Any]) -> None:
    """Flatten nested lora_adapters dicts into parallel arrays."""
    lora_adapters_raw = params.get("lora_adapters")
    if not lora_adapters_raw or not isinstance(lora_adapters_raw, list):
        return

    # Check if already in flattened format (list of strings)
    if lora_adapters_raw and isinstance(lora_adapters_raw[0], str):
        return

    # Nested dict format - flatten to parallel arrays
    lora_adapters = []
    lora_repos = []
    lora_filenames = []
    lora_scales = []

    for lora in lora_adapters_raw:
        if not isinstance(lora, dict):
            continue

        if "repo" in lora and "filename" in lora:
            lora_adapters.append("HF")
            lora_repos.append(lora["repo"])
            lora_filenames.append(lora["filename"])
        elif "path" in lora:
            lora_adapters.append(lora["path"])
        else:
            continue

        lora_scales.append(lora.get("scale", 1.0))

    params["lora_adapters"] = lora_adapters or [""]
    params["lora_adapters_repos"] = lora_repos or [""]
    params["lora_adapters_filenames"] = lora_filenames or [""]
    params["lora_adapters_scales"] = lora_scales or [0.0]


def process_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Process parameters: resolve special keys and extract metadata.

    Returns: (clean_params_dict, use_llava)
    """
    _resolve_lora_adapters(params)

    # Ensure stopping_words has a default
    if not params.get("stopping_words"):
        params["stopping_words"] = [""]

    # Extract use_llava (not a C++ node parameter)
    use_llava = params.pop("use_llava", False)

    return params, use_llava


def _create_node(
    params: Dict[str, Any],
    use_llava: bool,
    node_name: str = "",
    namespace: str = "llama",
) -> Node:
    """Create the appropriate ROS 2 Node action."""
    embedding = params.get("embedding", False)
    reranking = params.get("reranking", False)

    if use_llava:
        default_name = "llava_node"
        executable = "llava_node"
    elif embedding and not reranking:
        default_name = "llama_embedding_node"
        executable = "llama_node"
    elif reranking:
        default_name = "llama_reranking_node"
        executable = "llama_node"
    else:
        default_name = "llama_node"
        executable = "llama_node"

    return Node(
        package="llama_ros",
        executable=executable,
        name=node_name or default_name,
        namespace=namespace,
        parameters=[params],
    )


def create_llama_launch_from_yaml(
    file_path: str,
    node_name: str = "",
    namespace: str = "llama",
) -> Node:
    """Create a Node action from a YAML params file.

    Supports both ROS 2 params YAML format and legacy flat format.

    Args:
        file_path: Path to the model YAML file.
        node_name: Override the node name (default: auto-detected from model type).
        namespace: Override the namespace (default: "llama").
    """
    raw_params = load_params_from_yaml(file_path)
    params, use_llava = process_params(raw_params)
    return _create_node(params, use_llava, node_name=node_name, namespace=namespace)


def create_llama_launch(
    node_name: str = "",
    namespace: str = "llama",
    **kwargs,
) -> Node:
    """Create a Node action from keyword arguments.

    Args:
        node_name: Override the node name (default: auto-detected from model type).
        namespace: Override the namespace (default: "llama").
        **kwargs: Model parameters.
    """
    params, use_llava = process_params(dict(kwargs))
    return _create_node(params, use_llava, node_name=node_name, namespace=namespace)
