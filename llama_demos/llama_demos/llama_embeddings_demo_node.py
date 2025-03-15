#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024 Miguel Ángel González Santamarta
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


import sys
import rclpy
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.srv import GenerateEmbeddings


def main():
    if len(sys.argv) < 2:
        prompt = "This is the test to create embeddings using llama_ros"
    else:
        prompt = " ".join(sys.argv[1:])

    rclpy.init()

    llama_client = LlamaClientNode.get_instance()

    emb_req = GenerateEmbeddings.Request()
    emb_req.prompt = prompt

    emb = llama_client.generate_embeddings(emb_req).embeddings
    print(f"{emb}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
