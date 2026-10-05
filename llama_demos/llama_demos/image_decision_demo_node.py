#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Miguel Ángel González Santamarta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import urllib.request

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.msg import DecisionQuestion
from llama_msgs.srv import EvaluateDecisions

IMAGE_URL = "https://i.pinimg.com/474x/32/89/17/328917cc4fe3bd4cfbe2d32aa9cc6e98.jpg"


def load_image_from_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req)
    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def build_questions():
    choice = DecisionQuestion()
    choice.type = DecisionQuestion.CHOICE
    choice.instructions = "What is the race of the subject in the image?"
    choice.keys = ["human", "elf", "orc", "dwarf", "other"]

    score = DecisionQuestion()
    score.type = DecisionQuestion.SCORE
    score.instructions = "How big is the food?"
    score.descriptions = ["small", "medium", "large"]

    noul = DecisionQuestion()
    noul.type = DecisionQuestion.NOUL
    noul.instructions = "Is it a fantasy character?"

    return [choice, score, noul]


def print_answer(name, answer):
    if not answer.success:
        print(f"  {name}: failed: {answer.error}")
        return

    if answer.type == DecisionQuestion.CHOICE:
        print(f"  {name}: {answer.choice} (confidence {answer.confidence:.3f})")
    elif answer.type == DecisionQuestion.SCORE:
        print(f"  {name}: {answer.score:.2f} (confidence {answer.confidence:.3f})")
    else:
        print(f"  {name}: P(true) = {answer.noul:.3f}")

    for key, probability in zip(answer.keys, answer.probabilities):
        print(f"    {key}: {probability:.3f}")


def main():
    image_url = IMAGE_URL
    if len(sys.argv) > 1:
        image_url = sys.argv[1]

    rclpy.init()

    image = load_image_from_url(image_url)
    if image is None:
        print(f"Could not load image: {image_url}")
        rclpy.shutdown()
        return

    cv_bridge = CvBridge()
    image_msg = cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
    print(f"image: {image.shape[1]}x{image.shape[0]} from {image_url}")

    llama_client = LlamaClientNode.get_instance()

    req = EvaluateDecisions.Request()
    req.state = "A camera frame from the robot."
    req.images = [image_msg]
    req.questions = build_questions()

    response = llama_client.evaluate_decisions(req)

    if len(response.answers) != 3:
        print(f"expected 3 answers, got {len(response.answers)}")
        rclpy.shutdown()
        return

    print_answer("choice", response.answers[0])
    print_answer("score", response.answers[1])
    print_answer("noul", response.answers[2])

    rclpy.shutdown()


if __name__ == "__main__":
    main()
