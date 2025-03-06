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

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_ros.langchain import ChatLlamaROS


class ChatLlamaDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_llama_demo_node")

        self.declare_parameter("prompt", "Who is the character in the middle?")
        self.prompt = self.get_parameter("prompt").get_parameter_value().string_value

        self.cv_bridge = CvBridge()

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

    def send_prompt(self) -> None:

        self.chat = ChatLlamaROS(temp=0.2, penalty_last_n=8, template_method="jinja")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("You are an IA that answer questions."),
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"type": "text", "text": f"<image>{self.prompt}"},
                        {"type": "image_url", "image_url": "{image_url}"},
                    ]
                ),
            ]
        )

        self.chain = self.prompt | self.chat | StrOutputParser()

        self.initial_time = time.time()

        response = self.chain.invoke(
            {
                "image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg"
            }
        )
        
        print(response)


def main():
    rclpy.init()
    node = ChatLlamaDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
