#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from llama_msgs.srv import GPT


class LlamaClientNode(Node):

    def __init__(self) -> None:
        super().__init__("llama_client_node")

        self.declare_parameter(
            "prompt", "Do you know the city of León from Spain?\nCan you tell me a bit about its history?\n")
        self.prompt = self.get_parameter(
            "prompt").get_parameter_value().string_value
        self.prompt = self.prompt.replace("\\n", "\n")

        self.sub = self.create_subscription(
            String, "gpt_text", self.text_cb, 10)
        self.client = self.create_client(GPT, "gpt")

    def text_cb(self, msg: String) -> None:
        print(msg.data, end="", flush=True)

    def send_prompt(self) -> None:
        self.client.wait_for_service()
        req = GPT.Request()
        req.prompt = self.prompt
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        print("\n")


def main():

    rclpy.init()
    node = LlamaClientNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
