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


import cv2
import numpy as np
import requests

import rclpy
from cv_bridge import CvBridge
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.srv import GenerateEmbeddings
from std_msgs.msg import UInt8MultiArray

IMAGE_URL = "https://i.pinimg.com/474x/32/89/17/328917cc4fe3bd4cfbe2d32aa9cc6e98.jpg"
AUDIO_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/"
    "glass-breaking-151256.mp3"
)


def download_bytes(url: str) -> bytes:
    print(f"Downloading '{url}'")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.content


def main():
    rclpy.init()

    llama_client = LlamaClientNode.get_instance()
    cv_bridge = CvBridge()

    image = cv2.imdecode(
        np.frombuffer(download_bytes(IMAGE_URL), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    audio = download_bytes(AUDIO_URL)

    emb_req = GenerateEmbeddings.Request()
    emb_req.prompt = (
        "task: sentence similarity | query: describe the image and the sound "
        "<__media__> <__media__>"
    )
    emb_req.images.append(cv_bridge.cv2_to_imgmsg(image))

    audio_msg = UInt8MultiArray()
    audio_msg.data = list(audio)
    emb_req.audios.append(audio_msg)

    emb = llama_client.generate_embeddings(emb_req).embeddings
    print(f"Embedding size: {len(emb)}")
    print(f"First values: {emb}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
