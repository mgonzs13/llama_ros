#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024  Alejandro González Cantón
# Copyright (c) 2024  Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import time
from random import randint

import rclpy
from rclpy.node import Node

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from llama_ros.langchain import ChatLlamaROS


@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)


class ChatLlamaToolsDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chatllama_tools_demo_node")

        self.initial_time = -1
        self.tools_time = -1
        self.eval_time = -1

    def send_prompt(self) -> None:
        self.chat = ChatLlamaROS(temp=0.0, template_method="jinja")

        messages = [
            HumanMessage(
                "What is the current temperature in Madrid? And its inhabitants?"
            )
        ]

        self.get_logger().info(f"\nPrompt: {messages[0].content}")
        llm_tools = self.chat.bind_tools(
            [get_inhabitants, get_curr_temperature], tool_choice="any"
        )

        self.initial_time = time.time()
        all_tools_res = llm_tools.invoke(messages)
        self.tools_time = time.time()

        messages.append(all_tools_res)

        for tool in all_tools_res.tool_calls:
            selected_tool = {
                "get_inhabitants": get_inhabitants,
                "get_curr_temperature": get_curr_temperature,
            }[tool["name"]]

            tool_msg = selected_tool.invoke(tool)

            formatted_output = (
                f"{tool['name']}({''.join(tool['args'].values())}) = {tool_msg.content}"
            )
            self.get_logger().info(f"Calling tool: {formatted_output}")

            tool_msg.additional_kwargs = {"args": tool["args"]}
            messages.append(tool_msg)

        res = self.chat.invoke(messages)

        self.eval_time = time.time()

        self.get_logger().info(f"\nResponse: {res.content}")

        time_generate_tools = self.tools_time - self.initial_time
        time_last_response = self.eval_time - self.tools_time
        self.get_logger().info(f"Time to generate tools: {time_generate_tools} s")
        self.get_logger().info(f"Time to generate last response: {time_last_response} s")


def main():
    rclpy.init()
    node = ChatLlamaToolsDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
