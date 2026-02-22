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


from launch import LaunchDescription, LaunchContext
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from llama_bringup.utils import create_llama_launch_from_yaml


def generate_launch_description():

    def run_llama(context: LaunchContext, params_file, node_name, namespace):
        file_path = context.perform_substitution(params_file)
        name = context.perform_substitution(node_name)
        ns = context.perform_substitution(namespace)
        return [create_llama_launch_from_yaml(file_path, node_name=name, namespace=ns)]

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        description="Path to the model params YAML file for the LLM",
    )

    node_name = LaunchConfiguration("node_name")
    node_name_cmd = DeclareLaunchArgument(
        "node_name",
        default_value="",
        description="Override the node name (default: auto-detected from model type)",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="llama",
        description="Namespace for the node",
    )

    return LaunchDescription(
        [
            params_file_cmd,
            node_name_cmd,
            namespace_cmd,
            OpaqueFunction(function=run_llama, args=[params_file, node_name, namespace]),
        ]
    )
