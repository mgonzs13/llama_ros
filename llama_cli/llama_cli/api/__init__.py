# MIT License

# Copyright (c) 2024  Miguel Ángel González Santamarta

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


from launch import LaunchService
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch_from_yaml

import rclpy
from argparse import ArgumentTypeError
from llama_msgs.action import GenerateResponse
from llama_ros.llama_client_node import LlamaClientNode


def positive_float(inval):
    try:
        ret = float(inval)
    except ValueError:
        raise ArgumentTypeError("Expects a floating point number")
    if ret < 0.0:
        raise ArgumentTypeError("Value must be positive")
    return ret


def launch_llm(file_path: str) -> None:
    ld = LaunchDescription([
        create_llama_launch_from_yaml(file_path)
    ])
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


def prompt_llm(prompt: str, temp: float = 0.8) -> None:

    def text_cb(feedback) -> None:
        print(feedback.feedback.partial_response.text, end="", flush=True)

    rclpy.init()
    llama_client = LlamaClientNode()
    goal = GenerateResponse.Goal()
    goal.prompt = prompt
    goal.sampling_config.temp = temp
    llama_client.generate_response(goal, text_cb)
    rclpy.shutdown()
