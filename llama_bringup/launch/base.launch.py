
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
                "n_keep": LaunchConfiguration("n_keep", default=0),
                "top_k": LaunchConfiguration("top_k", default=40),

                "model": LaunchConfiguration("model", default=os.path.abspath(os.path.normpath(os.path.expanduser("~/llama_models/llama.bin")))),
                "prompt": LaunchConfiguration("prompt", default=""),
                "file": LaunchConfiguration("file", default=""),
                "input_prefix": LaunchConfiguration("input_prefix", default=""),

                "top_p": LaunchConfiguration("top_p", default=0.95),
                "temp": LaunchConfiguration("temp", default=0.80),
                "repeat_penalty": LaunchConfiguration("repeat_penalty", default=1.10),

                "memory_f16": LaunchConfiguration("memory_f16", default=True),
                "instruct": LaunchConfiguration("instruct", default=False),
                "ignore_eos": LaunchConfiguration("ignore_eos", default=False),
                "use_mlock": LaunchConfiguration("use_mlock", default=False),

                "reverse_prompt": LaunchConfiguration("reverse_prompt", default="['']")
            }]
        )
    ])
