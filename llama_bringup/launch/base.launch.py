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
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llama_node",
            name="llama_node",
            parameters=[{
                "seed": LaunchConfiguration("seed", default=-1),
                "n_ctx": LaunchConfiguration("n_ctx", default=512),
                "memory_f16": LaunchConfiguration("memory_f16", default=True),
                "use_mmap": LaunchConfiguration("use_mmap", default=True),
                "use_mlock": LaunchConfiguration("use_mlock", default=False),
                "embedding": LaunchConfiguration("embedding", default=True),

                "n_gpu_layers": LaunchConfiguration("n_gpu_layers", default=0),
                "main_gpu": LaunchConfiguration("main_gpu", default=0),
                "tensor_split": LaunchConfiguration("tensor_split", default="[0.0]"),
                "low_vram": LaunchConfiguration("tensor_split", default=False),

                "n_threads": LaunchConfiguration("n_threads", default=4),
                "n_predict": LaunchConfiguration("n_predict", default=128),
                "n_batch": LaunchConfiguration("n_batch", default=8),
                "n_keep": LaunchConfiguration("n_keep", default=-1),

                "temp": LaunchConfiguration("temp", default=0.80),
                "top_k": LaunchConfiguration("top_k", default=40),
                "top_p": LaunchConfiguration("top_p", default=0.95),
                "tfs_z": LaunchConfiguration("tfs_z", default=1.00),
                "typical_p": LaunchConfiguration("typical_p", default=1.00),
                "repeat_last_n": LaunchConfiguration("repeat_last_n", default=64),
                "repeat_penalty": LaunchConfiguration("repeat_penalty", default=1.10),
                "presence_penalty": LaunchConfiguration("presence_penalty", default=0.00),
                "frequency_penalty": LaunchConfiguration("frequency_penalty", default=0.00),
                "mirostat": LaunchConfiguration("mirostat", default=0),
                "mirostat_tau": LaunchConfiguration("mirostat_tau", default=5.00),
                "mirostat_eta": LaunchConfiguration("mirostat_eta", default=0.10),
                "penalize_nl": LaunchConfiguration("penalize_nl", default=True),

                "model": LaunchConfiguration("model", default=""),
                "lora_adapter": LaunchConfiguration("lora_adapter", default=""),
                "lora_base": LaunchConfiguration("lora_base", default=""),

                "prefix": ParameterValue(LaunchConfiguration("prefix", default=""), value_type=str),
                "suffix": ParameterValue(LaunchConfiguration("suffix", default=""), value_type=str),
                "stop": ParameterValue(LaunchConfiguration("stop", default=""), value_type=str),

                "prompt": ParameterValue(LaunchConfiguration("prompt", default=""), value_type=str),
                "file": LaunchConfiguration("file", default=""),
            }]
        )
    ])
