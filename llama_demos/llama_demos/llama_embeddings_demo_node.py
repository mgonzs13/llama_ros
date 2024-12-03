#!/usr/bin/env python3

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


import operator
import rclpy
from rclpy.node import Node
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.srv import GenerateEmbeddings


class LlamaEmbeddinsDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("llama_embeddings_demo_node")

        self.declare_parameter(
            "prompt", "This is the test to create embeddings using llama_ros"
        )
        self.prompt = self.get_parameter("prompt").get_parameter_value().string_value

        self._llama_client = LlamaClientNode.get_instance()

    def send_rerank(self) -> None:

        emb_req = GenerateEmbeddings.Request()
        emb_req.prompt = self.prompt

        emb = self._llama_client.generate_embeddings(emb_req).embeddings
        self.get_logger().info(f"{emb}")


def main():
    rclpy.init()
    node = LlamaEmbeddinsDemoNode()
    node.send_rerank()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
