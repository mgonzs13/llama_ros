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


from launch import LaunchService
from launch import LaunchDescription
from launch_ros.actions import Node

import os
import select
import signal
import sys
import time
from threading import Event
import yaml
import rclpy
from rclpy.signals import SignalHandlerOptions
from action_msgs.msg import GoalStatus
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


def _detect_executable(file_path: str) -> str:
    """Detect the ROS 2 executable from a params YAML file.

    Returns 'llava_node' if mmproj params are set, otherwise 'llama_node'.
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Navigate into ROS 2 params format
    params = {}
    if isinstance(data, dict):
        for key in data:
            inner = data[key]
            if isinstance(inner, dict) and "ros__parameters" in inner:
                params = inner["ros__parameters"]
                break

    mmproj = params.get("mmproj", {})
    if mmproj.get("repo") or mmproj.get("path"):
        return "llava_node"
    return "llama_node"


def launch_llm(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exists")
        return

    executable = _detect_executable(file_path)
    node = Node(
        package="llama_ros",
        executable=executable,
        name=executable,
        namespace="llama",
        parameters=[file_path],
    )
    ld = LaunchDescription([node])
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


def prompt_llm(
    prompt: str,
    reset: bool = False,
    temp: float = 0.8,
    image_url: str = "",
    *,
    precompute: bool = False,
    action_name: str = "/llama/generate_response",
    cancel_fd: int = None,
) -> int:
    """Run a response goal and cancel on a signal or cancellation-pipe EOF."""
    canceled = Event()
    previous_handlers = {}
    llama_client = None
    initialized = False
    request = None
    deadline = None
    streamed = []

    def cancel_signal(signum, frame):
        canceled.set()

    def cancellation_requested():
        if cancel_fd is not None and select.select([cancel_fd], [], [], 0)[0]:
            if not os.read(cancel_fd, 4096):
                canceled.set()
        return canceled.is_set()

    def feedback(message):
        if not precompute and not canceled.is_set():
            text = message.feedback.partial_response.text
            streamed.append(text)
            print(text, flush=True, end="")

    try:
        # Protect partial signal setup as well as the ROS request lifecycle.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, cancel_signal)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        initialized = True
        if cancellation_requested():
            return 1
        llama_client = LlamaClientNode(action_name=action_name)
        goal = GenerateResponse.Goal()
        goal.prompt = prompt
        goal.reset = reset
        goal.precompute = precompute
        goal.sampling_config.temp = temp

        if image_url:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            goal.images.append(CvBridge().cv2_to_imgmsg(img))

        while not cancellation_requested():
            if not rclpy.ok():
                return 1
            if llama_client.wait_for_response_server(timeout_sec=0.1):
                break
        if cancellation_requested():
            return 1
        request = llama_client.generate_response_async(goal, feedback_cb=feedback)
        while True:
            if cancellation_requested() or request.error is not None:
                if deadline is None:
                    deadline = time.monotonic() + 5.0
                    request.cancel()
                if time.monotonic() >= deadline:
                    print("Timed out waiting for action cancellation", file=sys.stderr)
                    return 1
            if request.done.is_set():
                break
            if not rclpy.ok():
                return 1
            request.done.wait(0.05)

        if canceled.is_set():
            return 1
        if request.error is not None:
            print(str(request.error), file=sys.stderr)
            return 1
        if request.status != GoalStatus.STATUS_SUCCEEDED:
            print(f"Action failed with status {request.status}", file=sys.stderr)
            return 1
        if not precompute:
            text = "".join(streamed)
            final_text = request.result.response.text
            if final_text.startswith(text):
                print(final_text[len(text) :], flush=True, end="")
                text = final_text
            if not text.endswith("\n"):
                print()
        return 0
    except Exception as exc:
        if request is not None and not request.done.is_set():
            request.cancel()
        print(f"Prompt failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if llama_client is not None:
                llama_client.close()
        finally:
            try:
                if initialized:
                    rclpy.shutdown()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
