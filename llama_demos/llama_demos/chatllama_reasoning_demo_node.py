#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024 Alejandro González Cantón
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


import time
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from llama_ros.langchain import ChatLlamaROS

class ChatLlamaReasoningDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_llama_demo_node")

        self.declare_parameter("prompt", "Here we have a book, a laptop and a nail. Please tell me how to stack them onto each other in a stable manner.")
        self.str_prompt = self.get_parameter("prompt").get_parameter_value().string_value

        self.cv_bridge = CvBridge()

        self.initial_time = -1
        self.eval_time = -1

    def send_prompt(self) -> None:

        self.chat = ChatLlamaROS(temp=0.2, penalty_last_n=8)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"type": "text", "text": f"{self.str_prompt}"},
                    ]
                ),
            ]
        )

        self.chain = self.prompt | self.chat

        self.initial_time = time.time()
        response = self.chain.invoke({})
        self.final_time = time.time()

        print(f'Prompt: {self.str_prompt}')
        print(f'Response: {response.content.strip()}')
        print(f'Reasoning char size: {len(response.additional_kwargs["reasoning_content"])}')

        print(f"Time elapsed: {self.final_time - self.initial_time:.2f} seconds")

def main():
    rclpy.init()
    node = ChatLlamaReasoningDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
