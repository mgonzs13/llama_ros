#!/usr/bin/env python3

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


import time
import cv2
import numpy as np
import urllib.request

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse


class LlavaDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("llava_demo_node")

        self.cv_bridge = CvBridge()

        self.declare_parameter(
            "prompt", "Who is the character in the middle of the image?"
        )
        self.prompt = self.get_parameter("prompt").get_parameter_value().string_value

        self.declare_parameter("use_image", True)
        self.use_image = (
            self.get_parameter("use_image").get_parameter_value().bool_value
        )

        self.declare_parameter(
            "image_url",
            "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg",
        )
        self.image = self.load_image_from_url(
            self.get_parameter("image_url").get_parameter_value().string_value
        )

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

        self._llama_client = LlamaClientNode.get_instance()

    @staticmethod
    def load_image_from_url(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img

    def text_cb(self, feedback) -> None:

        if self.eval_time < 0:
            self.eval_time = time.time()

        self.tokens += 1
        print(feedback.feedback.partial_response.text, end="", flush=True)

    def send_prompt(self) -> None:

        goal = GenerateResponse.Goal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2
        goal.sampling_config.penalty_last_n = 8

        if self.use_image:
            goal.image = self.cv_bridge.cv2_to_imgmsg(self.image)

        self.initial_time = time.time()
        self._llama_client.generate_response(goal, self.text_cb)

        self.get_logger().info("END")
        end_time = time.time()
        self.get_logger().info(f"Time to eval: {self.eval_time - self.initial_time} s")
        self.get_logger().info(
            f"Prediction speed: {self.tokens / (end_time - self.eval_time)} t/s"
        )


def main():
    rclpy.init()
    node = LlavaDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
