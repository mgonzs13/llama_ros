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
                "n_ctx": "512",

                "n_threads": "4",
                "n_predict": "512",
                "n_batch": "8",

                "temp": "0.2",
                "top_k": "40",
                "repeat_last_n": "8",

                "model": os.path.abspath(os.path.normpath(os.path.expanduser("~/llama_models/wizard.bin"))),

                "prefix": "\n\n### Instruction:\n",
                "suffix": "\n\n### Response:\n",
                "stop": "### Instruction:\n",

                "file": os.path.join(llama_bringup_shared_dir, "prompts/alpaca.txt"),
            }.items()
        )
    ])
