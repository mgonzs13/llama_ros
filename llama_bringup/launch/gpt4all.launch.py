
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    llama_bringup_shared_dir = get_package_share_directory(
        "llama_bringup")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(llama_bringup_shared_dir, "launch", "base.launch.py")),
            launch_arguments={
                "n_threads": "4",
                "n_predict": "512",
                "repeat_last_n": "8",
                "n_ctx": "512",
                "n_batch": "512",
                "keep": "0",
                "top_k": "64",

                "model": os.path.abspath(os.path.normpath(os.path.expanduser("~/llama_models/gpt4all.bin"))),
                "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",

                "temp": "0.2",

                "instruct": "True"
            }.items()
        )
    ])
