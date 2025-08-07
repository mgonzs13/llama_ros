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


from typing import List
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    def run_llama(context: LaunchContext, embedding, reranking):
        embedding = eval(context.perform_substitution(embedding))
        reranking = eval(context.perform_substitution(reranking))

        params = {
            "verbosity": LaunchConfiguration("verbosity", default=0),
            "seed": LaunchConfiguration("seed", default=-1),
            "n_ctx": LaunchConfiguration("n_ctx", default=512),
            "n_batch": LaunchConfiguration("n_batch", default=2048),
            "n_ubatch": LaunchConfiguration("n_batch", default=512),
            # GPU params
            "devices": LaunchConfiguration("devices", default="['']"),
            "n_gpu_layers": LaunchConfiguration("n_gpu_layers", default=0),
            "split_mode": LaunchConfiguration("split_mode", default="layer"),
            "main_gpu": LaunchConfiguration("main_gpu", default=0),
            "tensor_split": LaunchConfiguration("tensor_split", default="[0.0]"),
            # attn params
            "grp_attn_n": LaunchConfiguration("grp_attn_n", default=1),
            "grp_attn_w": LaunchConfiguration("grp_attn_w", default=512),
            # rope params
            "rope_freq_base": LaunchConfiguration("rope_freq_base", default=0.0),
            "rope_freq_scale": LaunchConfiguration("rope_freq_scale", default=0.0),
            "rope_scaling_type": LaunchConfiguration("rope_scaling_type", default=""),
            # yarn params
            "yarn_ext_factor": LaunchConfiguration("yarn_ext_factor", default=-1.0),
            "yarn_attn_factor": LaunchConfiguration("yarn_attn_factor", default=1.0),
            "yarn_beta_fast": LaunchConfiguration("yarn_beta_fast", default=32.0),
            "yarn_beta_slow": LaunchConfiguration("yarn_beta_slow", default=1.0),
            "yarn_orig_ctx": LaunchConfiguration("yarn_orig_ctx", default=0),
            "defrag_thold": LaunchConfiguration("defrag_thold", default=0.1),
            # bool params
            "embedding": embedding,
            "reranking": reranking,
            "use_mmap": LaunchConfiguration("use_mmap", default=True),
            "use_mlock": LaunchConfiguration("use_mlock", default=False),
            "warmup": LaunchConfiguration("warmup", default=True),
            "check_tensors": LaunchConfiguration("check_tensors", default=False),
            "flash_attn": LaunchConfiguration("flash_attn", default=False),
            # cache params
            "no_op_offload": LaunchConfiguration("no_op_offload", default=False),
            "no_extra_bufts": LaunchConfiguration("no_extra_bufts", default=False),
            "no_kv_offload": LaunchConfiguration("no_kv_offload", default=False),
            "cache_type_k": LaunchConfiguration("cache_type_k", default="f16"),
            "cache_type_v": LaunchConfiguration("cache_type_v", default="f16"),
            # CPU params
            "n_threads": LaunchConfiguration("n_threads", default=1),
            "cpu_mask": LaunchConfiguration("cpu_mask", default=""),
            "cpu_range": LaunchConfiguration("cpu_range", default=""),
            "priority": LaunchConfiguration("priority", default="normal"),
            "strict_cpu": LaunchConfiguration("strict_cpu", default=False),
            "poll": LaunchConfiguration("poll", default=50),
            # batch CPU params
            "n_threads_batch": LaunchConfiguration("n_threads_batch", default=1),
            "cpu_mask_batch": LaunchConfiguration("cpu_mask_batch", default=""),
            "cpu_range_batch": LaunchConfiguration("cpu_range_batch", default=""),
            "priority_batch": LaunchConfiguration("priority_batch", default="normal"),
            "strict_cpu_batch": LaunchConfiguration("strict_cpu_batch", default=False),
            "poll_batch": LaunchConfiguration("poll_batch", default=50),
            # switch context params
            "n_predict": LaunchConfiguration("n_predict", default=128),
            "n_keep": LaunchConfiguration("n_keep", default=-1),
            # multimodal params
            "mmproj_use_gpu": LaunchConfiguration("mmproj_use_gpu", default=True),
            "no_mmproj": LaunchConfiguration("no_mmproj", default=False),
            # paths params
            "model_path": LaunchConfiguration("model_path", default=""),
            "model_repo": LaunchConfiguration("model_repo", default=""),
            "model_filename": LaunchConfiguration("model_filename", default=""),
            "mmproj_path": LaunchConfiguration("mmproj_path", default=""),
            "mmproj_repo": LaunchConfiguration("mmproj_repo", default=""),
            "mmproj_filename": LaunchConfiguration("mmproj_filename", default=""),
            "lora_adapters": LaunchConfiguration("lora_adapters", default="['']"),
            "lora_adapters_repos": LaunchConfiguration(
                "lora_adapters_repos", default="['']"
            ),
            "lora_adapters_filenames": LaunchConfiguration(
                "lora_adapters_filenames", default="['']"
            ),
            "lora_adapters_scales": LaunchConfiguration(
                "lora_adapters_scales", default="[0.0]"
            ),
            "numa": LaunchConfiguration("numa", default="none"),
            "pooling_type": LaunchConfiguration("pooling_type", default=""),
            # prefix/suffix
            "prefix": ParameterValue(
                LaunchConfiguration("prefix", default=""), value_type=str
            ),
            "suffix": ParameterValue(
                LaunchConfiguration("suffix", default=""), value_type=str
            ),
            "stopping_words": LaunchConfiguration("stopping_words", default="['']"),
            "chat_template_file": LaunchConfiguration("chat_template_file", default=""),
            # prompt params
            "system_prompt": ParameterValue(
                LaunchConfiguration("system_prompt", default=""), value_type=str
            ),
            "system_prompt_file": ParameterValue(
                LaunchConfiguration("system_prompt_file", default=""), value_type=str
            ),
        }

        # get llama node name
        llama_node_name = "llama_node"

        if embedding and not reranking:
            llama_node_name = "llama_embedding_node"
        elif reranking:
            llama_node_name = "llama_reranking_node"

        return Node(
            package="llama_ros",
            executable="llama_node",
            name=llama_node_name,
            namespace="llama",
            parameters=[params],
            condition=UnlessCondition(
                PythonExpression([LaunchConfiguration("use_llava")])
            ),
        ), Node(
            package="llama_ros",
            executable="llava_node",
            name="llava_node",
            namespace="llama",
            parameters=[params],
            condition=IfCondition(PythonExpression([LaunchConfiguration("use_llava")])),
        )

    embedding = LaunchConfiguration("embedding")
    embedding_cmd = DeclareLaunchArgument(
        "embedding",
        default_value="False",
        description="Whether the model is an embedding model",
    )

    reranking = LaunchConfiguration("reranking")
    reranking_cmd = DeclareLaunchArgument(
        "reranking",
        default_value="False",
        description="Whether the model is an reranking model",
    )

    return LaunchDescription(
        [
            embedding_cmd,
            reranking_cmd,
            OpaqueFunction(function=run_llama, args=[embedding, reranking]),
        ]
    )
