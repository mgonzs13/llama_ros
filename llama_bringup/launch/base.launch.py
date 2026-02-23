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


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        description="Path to the ROS 2 params YAML file for the model",
    )

    executable = LaunchConfiguration("executable")
    executable_cmd = DeclareLaunchArgument(
        "executable",
        default_value="llama_node",
        description="Executable to run (llama_node or llava_node)",
    )

    node_name = LaunchConfiguration("node_name")
    node_name_cmd = DeclareLaunchArgument(
        "node_name",
        default_value="llama_node",
        description="Name for the node",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="llama",
        description="Namespace for the node",
    )

    llama_node = Node(
        package="llama_ros",
        executable=executable,
        name=node_name,
        namespace=namespace,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            params_file_cmd,
            executable_cmd,
            node_name_cmd,
            namespace_cmd,
            llama_node,
        ]
    )
