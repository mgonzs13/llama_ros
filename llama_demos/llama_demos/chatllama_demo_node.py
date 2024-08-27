#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024  Alejandro González Cantón
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
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ChatLlamaDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_llama_demo_node")

        self.cv_bridge = CvBridge()

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

        self._llama_client = LlamaClientNode.get_instance()

    @staticmethod
    def load_image_from_url(url):
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img

    def send_prompt(self) -> None:

        self.chat = ChatLlamaROS(
            temp=0.2,
            penalty_last_n=8,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    "You are a IA that just answer with a single word."),
                HumanMessage(content=[
                    {"type": "text", "text": "<image>\nWhat is the character in the middle of the image?"},
                    {"type": "image_url", "image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg"}
                ])
            ]
        )

        self.chain = self.prompt | self.chat | StrOutputParser()

        self.initial_time = time.time()

        self.response = self.chain.invoke({})
        self.get_logger().info(self.response)

        self.get_logger().info("END")


def main():
    rclpy.init()
    node = ChatLlamaDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
