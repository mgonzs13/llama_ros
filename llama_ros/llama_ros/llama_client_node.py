#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from llama_msgs.action import GPT


class LlamaClientNode(Node):

    def __init__(self) -> None:
        super().__init__("llama_client_node")

        self.declare_parameter(
            "prompt", "Do you know the city of León from Spain?\nCan you tell me a bit about its history?")
        self.prompt = self.get_parameter(
            "prompt").get_parameter_value().string_value
        self.prompt = self.prompt.replace("\\n", "\n")

        self._get_result_future = None
        self._action_client = ActionClient(self, GPT, "gpt")

    def text_cb(self, msg) -> None:
        print(msg.feedback.text, end="", flush=True)

    def send_prompt(self) -> None:

        goal = GPT.Goal()
        goal.prompt = self.prompt

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=self.text_cb)

        rclpy.spin_until_future_complete(self, send_goal_future)
        get_result_future = send_goal_future.result().get_result_async()

        rclpy.spin_until_future_complete(self, get_result_future)
        # result = get_result_future.result().result

        self.get_logger().info("END")


def main():

    rclpy.init()
    node = LlamaClientNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
