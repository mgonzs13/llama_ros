
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
                "n_predict": "256",
                "repeat_last_n": "8",
                "n_ctx": "512",
                "n_batch": "1024",
                "keep": "48",
                "top_k": "64",

                "model": os.path.abspath(os.path.normpath(os.path.expanduser("~/llama_models/llama.bin"))),
                "file": os.path.join(llama_bringup_shared_dir, "prompts/chat-with-bob.txt"),

                "prefix": "\n\n### User:\n",
                "suffix": "\n\n### Bob:\n",
                "stop": "### User:\n",

                "temp": "0.8",
            }.items()
        )
    ])
