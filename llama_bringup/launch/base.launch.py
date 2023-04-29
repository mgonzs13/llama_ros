
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
                "n_threads": LaunchConfiguration("n_threads", default=4),
                "n_predict": LaunchConfiguration("n_predict", default=128),
                "repeat_last_n": LaunchConfiguration("repeat_last_n", default=64),
                "n_parts": LaunchConfiguration("n_parts", default=-1),
                "n_ctx": LaunchConfiguration("n_ctx", default=512),
                "n_batch": LaunchConfiguration("n_batch", default=8),
                "n_keep": LaunchConfiguration("n_keep", default=-1),
                "top_k": LaunchConfiguration("top_k", default=40),
                "mirostat": LaunchConfiguration("mirostat", default=0),

                "model": LaunchConfiguration("model", default=os.path.abspath(os.path.normpath(os.path.expanduser("~/llama_models/llama.bin")))),
                "lora_adapter": LaunchConfiguration("lora_adapter", default=""),
                "lora_base": LaunchConfiguration("lora_base", default=""),
                "prompt": ParameterValue(LaunchConfiguration("prompt", default=""), value_type=str),
                "file": LaunchConfiguration("file", default=""),
                "prefix": ParameterValue(LaunchConfiguration("prefix", default=""), value_type=str),
                "suffix": ParameterValue(LaunchConfiguration("suffix", default=""), value_type=str),
                "stop": ParameterValue(LaunchConfiguration("stop", default=""), value_type=str),

                "temp": LaunchConfiguration("temp", default=0.80),
                "top_p": LaunchConfiguration("top_p", default=0.95),
                "tfs_z": LaunchConfiguration("tfs_z", default=1.00),
                "typical_p": LaunchConfiguration("typical_p", default=1.00),
                "presence_penalty": LaunchConfiguration("presence_penalty", default=0.00),
                "frequency_penalty": LaunchConfiguration("frequency_penalty", default=0.00),
                "mirostat_tau": LaunchConfiguration("mirostat_tau", default=5.10),
                "mirostat_eta": LaunchConfiguration("mirostat_eta", default=0.10),
                "repeat_penalty": LaunchConfiguration("repeat_penalty", default=1.10),

                "memory_f16": LaunchConfiguration("memory_f16", default=True),
                "use_mmap": LaunchConfiguration("use_mmap", default=True),
                "use_mlock": LaunchConfiguration("use_mlock", default=False),
                "embedding": LaunchConfiguration("embedding", default=True),
                "penalize_nl": LaunchConfiguration("penalize_nl", default=True),
            }]
        )
    ])
