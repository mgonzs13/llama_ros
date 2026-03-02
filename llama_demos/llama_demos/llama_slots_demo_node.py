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


import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from llama_msgs.action import GenerateResponse


PROMPTS = [
    "Tell me a brief fun fact about cats.",
    "Tell me a brief fun fact about dogs.",
    "Tell me a brief fun fact about birds.",
    "Tell me a brief fun fact about fish.",
]


class SlotsDemoNode(Node):

    def __init__(self):
        super().__init__("slots_demo_node", namespace="llama")

        self._cb_group = ReentrantCallbackGroup()
        self._action_client = ActionClient(
            self,
            GenerateResponse,
            "generate_response",
            callback_group=self._cb_group,
        )

        self._print_lock = threading.Lock()

    def send_request(self, slot_idx, prompt, results):
        self._action_client.wait_for_server()

        goal = GenerateResponse.Goal()
        goal.prompt = prompt
        goal.sampling_config.temp = 0.2

        tokens = 0
        initial_time = time.time()
        eval_time = -1
        done_event = threading.Event()
        result_data = {}

        def feedback_cb(feedback):
            nonlocal eval_time, tokens
            if eval_time < 0:
                eval_time = time.time()
            tokens += 1
            text = feedback.feedback.partial_response.text
            with self._print_lock:
                print(f"[Slot {slot_idx}] {text}", end="", flush=True)

        def result_cb(future):
            nonlocal result_data
            end_time = time.time()
            result_data = {
                "tokens": tokens,
                "time_to_eval": (eval_time - initial_time if eval_time > 0 else 0),
                "speed": (
                    tokens / (end_time - eval_time) if eval_time > 0 and tokens > 0 else 0
                ),
                "total_time": end_time - initial_time,
            }
            done_event.set()

        def goal_response_cb(future):
            goal_handle = future.result()
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(result_cb)

        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback_cb
        )
        send_goal_future.add_done_callback(goal_response_cb)

        done_event.wait()
        results[slot_idx] = result_data


def main():
    rclpy.init()
    node = SlotsDemoNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    n_parallel = len(PROMPTS)
    print(f"\nSending {n_parallel} concurrent requests " f"to test parallel slots...\n")

    results = {}
    threads = []

    initial_time = time.time()

    for i, prompt in enumerate(PROMPTS):
        t = threading.Thread(target=node.send_request, args=(i, prompt, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time = time.time() - initial_time

    print(f"\n\n{'='*60}")
    print(f"Parallel Slots Demo Results ({n_parallel} slots)")
    print(f"{'='*60}")

    total_tokens = 0
    for i in sorted(results.keys()):
        r = results[i]
        total_tokens += r["tokens"]
        print(
            f"Slot {i}: {r['tokens']} tokens, "
            f"time to eval: {r['time_to_eval']:.4f}s, "
            f"speed: {r['speed']:.2f} t/s, "
            f"total: {r['total_time']:.4f}s"
        )

    print(f"{'='*60}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Aggregate throughput: {total_tokens / total_time:.2f} t/s")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
