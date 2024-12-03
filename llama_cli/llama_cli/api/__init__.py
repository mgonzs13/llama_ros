# MIT License
#
# Copyright (c) 2024  Miguel Ángel González Santamarta
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


from launch import LaunchService
from launch import LaunchDescription
from llama_bringup.utils import create_llama_launch_from_yaml

import os
import rclpy
from argparse import ArgumentTypeError
from llama_msgs.action import GenerateResponse
from llama_ros.llama_client_node import LlamaClientNode

import cv2
import numpy as np
import urllib.request
from cv_bridge import CvBridge


def positive_float(inval):
    try:
        ret = float(inval)
    except ValueError:
        raise ArgumentTypeError("Expects a floating point number")
    if ret < 0.0:
        raise ArgumentTypeError("Value must be positive")
    return ret


def launch_llm(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exists")
        return

    ld = LaunchDescription([create_llama_launch_from_yaml(file_path)])
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


def prompt_llm(
    prompt: str, reset: bool = False, temp: float = 0.8, image_url: str = ""
) -> None:

    rclpy.init()
    llama_client = LlamaClientNode()
    goal = GenerateResponse.Goal()
    goal.prompt = prompt
    goal.reset = reset
    goal.sampling_config.temp = temp

    if image_url:
        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        cv_bridge = CvBridge()
        goal.image = cv_bridge.cv2_to_imgmsg(img)

    last_t = ""
    for ele in llama_client.generate_response(goal, stream=True):
        last_t = ele.text
        print(ele.text, flush=True, end="")
    if not last_t.endswith("\n"):
        print()
    rclpy.shutdown()
