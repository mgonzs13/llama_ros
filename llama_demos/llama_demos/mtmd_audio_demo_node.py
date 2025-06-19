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
import numpy as np
import requests
import tempfile


import rclpy
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse
from std_msgs.msg import UInt8MultiArray


def download_audio_to_tempfile(url: str) -> str:
    """Download WAV file to a temporary file and return its path."""
    response = requests.get(url)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name


def read_mp3_as_uint8_array(filename: str) -> np.ndarray:
    """Read the binary MP3 file and return a NumPy array of uint8."""
    with open(filename, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)


def text_cb(feedback):
    global eval_time, tokens
    if eval_time < 0:
        eval_time = time.time()
    tokens += 1
    print(feedback.feedback.partial_response.text, end="", flush=True)


def main():
    prompt = "<__media__>What's that sound?"
    use_audio = True
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    if len(sys.argv) > 2:
        use_audio = sys.argv[2].lower() in ["true", "1", "yes"]
    if len(sys.argv) > 3:
        use_audio = sys.argv[3]

    global tokens, eval_time
    tokens = 0
    eval_time = -1

    rclpy.init()
    file_path = download_audio_to_tempfile(audio_url)
    mp3_array = read_mp3_as_uint8_array(file_path)
    llama_client = LlamaClientNode.get_instance()

    goal = GenerateResponse.Goal()
    goal.prompt = prompt
    goal.sampling_config.temp = 0.8

    if use_audio and mp3_array is not None:
        msg = UInt8MultiArray()
        msg.data = mp3_array.tolist()
        goal.audios.append(msg)

    initial_time = time.time()
    llama_client.generate_response(goal, text_cb)
    end_time = time.time()

    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
