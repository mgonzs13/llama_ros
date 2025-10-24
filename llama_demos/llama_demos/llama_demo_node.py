#!/usr/bin/env python3

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


import sys
import time
import rclpy
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse


def text_cb(feedback):
    global eval_time, tokens
    if eval_time < 0:
        eval_time = time.time()
    tokens += 1
    print(feedback.feedback.partial_response.text, end="", flush=True)


def main():
    if len(sys.argv) < 2:
        prompt = "Do you know the city of León from Spain? Can you tell me a bit about its history?"
    else:
        prompt = " ".join(sys.argv[1:])

    global tokens, eval_time
    tokens = 0
    eval_time = -1

    rclpy.init()
    llama_client = LlamaClientNode.get_instance()

    goal = GenerateResponse.Goal()
    goal.prompt = prompt
    goal.sampling_config.temp = 0.2

    initial_time = time.time()
    llama_client.generate_response(goal, text_cb)
    end_time = time.time()

    print(f"\nTime to eval: {eval_time - initial_time:.4f} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time):.4f} t/s")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
