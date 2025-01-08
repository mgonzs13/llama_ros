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

import rclpy
from rclpy.node import Node
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from random import randint


@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return 7100000

@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 40)


class ChatLlamaToolsDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_tools_demo_node")

        self.declare_parameter(
            "prompt", "What is the temperature today in leon?")
        self.prompt = self.get_parameter(
            "prompt").get_parameter_value().string_value

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

    def send_prompt(self) -> None:
        self.chat = ChatLlamaROS(
            temp=0.6,
            penalty_last_n=8,
            use_llama_template=True
        )

        messages = [
            # SystemMessage("You are an IA that solves problems. You ouput in JSON format. The key 'tool_calls' is a list of possible tools, like 'get_inhabitants' or 'get_max_temperature'. For each tool, the format is {{name, arguments}}"),
            HumanMessage("What is the current temperature in Madrid? And its inhabitants?")
        ]
                
        llm_tools = self.chat.bind_tools([get_inhabitants, get_curr_temperature], tool_choice='any')
        
        all_tools_res = llm_tools.invoke(messages)
        
        messages.append(all_tools_res)
        
        for tool in all_tools_res.tool_calls:
            selected_tool = {"get_inhabitants": get_inhabitants, "get_curr_temperature": get_curr_temperature}[tool['name']]
            tool_msg = selected_tool.invoke(tool)
            tool_msg.additional_kwargs = {'args': tool['args']}
            messages.append(tool_msg)
        
        res = self.chat.invoke(messages)
        self.get_logger().info(res.content)


def main():
    rclpy.init()
    node = ChatLlamaToolsDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
