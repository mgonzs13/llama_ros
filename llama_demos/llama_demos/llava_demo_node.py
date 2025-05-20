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
import time
import cv2
import numpy as np
import urllib.request

import rclpy
from cv_bridge import CvBridge
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse


def load_image_from_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req)
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def text_cb(feedback):
    global eval_time, tokens
    if eval_time < 0:
        eval_time = time.time()
    tokens += 1
    print(feedback.feedback.partial_response.text, end="", flush=True)


def main():
    prompt = "<__image__>What type of food is the girl holding?"
    use_image = True
    image_url = "https://i.pinimg.com/474x/32/89/17/328917cc4fe3bd4cfbe2d32aa9cc6e98.jpg"

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    if len(sys.argv) > 2:
        use_image = sys.argv[2].lower() in ["true", "1", "yes"]
    if len(sys.argv) > 3:
        image_url = sys.argv[3]

    global tokens, eval_time
    tokens = 0
    eval_time = -1

    rclpy.init()
    cv_bridge = CvBridge()
    image = load_image_from_url(image_url) if use_image else None
    llama_client = LlamaClientNode.get_instance()

    goal = GenerateResponse.Goal()
    goal.prompt = prompt
    goal.sampling_config.temp = 0.2

    if use_image and image is not None:
        goal.images.append(cv_bridge.cv2_to_imgmsg(image))

    initial_time = time.time()
    llama_client.generate_response(goal, text_cb)
    end_time = time.time()

    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
