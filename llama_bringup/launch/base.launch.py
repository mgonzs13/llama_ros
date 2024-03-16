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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    params = {
        "seed": LaunchConfiguration("seed", default=-1),
        "n_ctx": LaunchConfiguration("n_ctx", default=512),
        "n_batch": LaunchConfiguration("n_batch", default=2048),
        "n_ubatch": LaunchConfiguration("n_batch", default=512),

        "n_gpu_layers": LaunchConfiguration("n_gpu_layers", default=0),
        "split_mode": LaunchConfiguration("split_mode", default="layer"),
        "main_gpu": LaunchConfiguration("main_gpu", default=0),
        "tensor_split": LaunchConfiguration("tensor_split", default="[0.0]"),

        "grp_attn_n": LaunchConfiguration("grp_attn_n", default=1),
        "grp_attn_w": LaunchConfiguration("grp_attn_w", default=512),

        "rope_freq_base": LaunchConfiguration("rope_freq_base", default=0.0),
        "rope_freq_scale": LaunchConfiguration("rope_freq_scale", default=0.0),
        "rope_scaling_type": LaunchConfiguration("rope_scaling_type", default=""),

        "yarn_ext_factor": LaunchConfiguration("yarn_ext_factor", default=-1.0),
        "yarn_attn_factor": LaunchConfiguration("yarn_attn_factor", default=1.0),
        "yarn_beta_fast": LaunchConfiguration("yarn_beta_fast", default=32.0),
        "yarn_beta_slow": LaunchConfiguration("yarn_beta_slow", default=1.0),
        "yarn_orig_ctx": LaunchConfiguration("yarn_orig_ctx", default=0),

        "embedding": LaunchConfiguration("embedding", default=True),
        "logits_all": LaunchConfiguration("logits_all", default=False),
        "use_mmap": LaunchConfiguration("use_mmap", default=True),
        "use_mlock": LaunchConfiguration("use_mlock", default=False),

        "dump_kv_cache": LaunchConfiguration("dump_kv_cache", default=False),
        "no_kv_offload": LaunchConfiguration("no_kv_offload", default=False),
        "cache_type_k": LaunchConfiguration("cache_type_k", default="f16"),
        "cache_type_v": LaunchConfiguration("cache_type_v", default="f16"),

        "n_threads": LaunchConfiguration("n_threads", default=4),
        "n_threads_batch": LaunchConfiguration("n_threads_batch", default=-1),
        "n_predict": LaunchConfiguration("n_predict", default=128),
        "n_keep": LaunchConfiguration("n_keep", default=-1),

        "model": LaunchConfiguration("model", default=""),
        "lora_adapter": LaunchConfiguration("lora_adapter", default=""),
        "lora_base": LaunchConfiguration("lora_base", default=""),
        "mmproj": LaunchConfiguration("mmproj", default=""),
        "numa": LaunchConfiguration("numa", default="none"),
        "pooling_type": LaunchConfiguration("pooling_type", default=""),

        "prefix": ParameterValue(LaunchConfiguration("prefix", default=""), value_type=str),
        "suffix": ParameterValue(LaunchConfiguration("suffix", default=""), value_type=str),
        "stop": ParameterValue(LaunchConfiguration("stop", default=""), value_type=str),

        "prompt": ParameterValue(LaunchConfiguration("prompt", default=""), value_type=str),
        "file": LaunchConfiguration("file", default=""),
        "debug": LaunchConfiguration("debug", default=True),
    }

    return LaunchDescription([
        Node(
            package="llama_ros",
            executable="llama_node",
            name="llama_node",
            namespace="llama",
            parameters=[params],
            condition=UnlessCondition(PythonExpression(
                [LaunchConfiguration("use_llava")]))
        ),

        Node(
            package="llama_ros",
            executable="llava_node",
            name="llava_node",
            namespace="llava",
            parameters=[params],
            condition=IfCondition(PythonExpression(
                [LaunchConfiguration("use_llava")]))
        ),
    ])
