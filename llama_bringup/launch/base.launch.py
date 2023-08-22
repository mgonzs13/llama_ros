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
            namespace="llama",
            parameters=[{
                "seed": LaunchConfiguration("seed", default=-1),
                "n_ctx": LaunchConfiguration("n_ctx", default=512),
                "n_batch": LaunchConfiguration("n_batch", default=8),

                "n_gpu_layers": LaunchConfiguration("n_gpu_layers", default=0),
                "main_gpu": LaunchConfiguration("main_gpu", default=0),
                "tensor_split": LaunchConfiguration("tensor_split", default="[0.0]"),

                "rope_freq_base": LaunchConfiguration("rope_freq_base", default=10000.0),
                "rope_freq_scale": LaunchConfiguration("rope_freq_scale", default=1.0),

                "low_vram": LaunchConfiguration("low_vram", default=False),
                "mul_mat_q": LaunchConfiguration("mul_mat_q", default=False),
                "f16_kv": LaunchConfiguration("f16_kv", default=True),
                "logits_all": LaunchConfiguration("logits_all", default=False),
                "vocab_only": LaunchConfiguration("vocab_only", default=False),
                "use_mmap": LaunchConfiguration("use_mmap", default=True),
                "use_mlock": LaunchConfiguration("use_mlock", default=False),
                "embedding": LaunchConfiguration("embedding", default=True),

                "n_threads": LaunchConfiguration("n_threads", default=4),
                "n_predict": LaunchConfiguration("n_predict", default=128),
                "n_keep": LaunchConfiguration("n_keep", default=-1),

                "model": LaunchConfiguration("model", default=""),
                "lora_adapter": LaunchConfiguration("lora_adapter", default=""),
                "lora_base": LaunchConfiguration("lora_base", default=""),
                "numa": LaunchConfiguration("numa", default=True),

                "prefix": ParameterValue(LaunchConfiguration("prefix", default=""), value_type=str),
                "suffix": ParameterValue(LaunchConfiguration("suffix", default=""), value_type=str),
                "stop": ParameterValue(LaunchConfiguration("stop", default=""), value_type=str),

                "prompt": ParameterValue(LaunchConfiguration("prompt", default=""), value_type=str),
                "file": LaunchConfiguration("file", default=""),
            }]
        )
    ])
