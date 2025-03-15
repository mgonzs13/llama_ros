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
from langchain_core.messages import AIMessage
from llama_ros.langchain import ChatLlamaROS
from typing import Optional

from pydantic import BaseModel, Field


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ChatLlamaStructuredDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("chat_llama_demo_node")

        self.cv_bridge = CvBridge()

        self.tokens = 0
        self.initial_time = -1
        self.eval_time = -1

    def send_prompt(self) -> None:

        self.chat = ChatLlamaROS(temp=0.2, penalty_last_n=8)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"type": "text", "text": "{prompt}"},
                    ]
                ),
            ]
        )

        structured_chat = self.chat.with_structured_output(
            Joke, method="function_calling"
        )

        self.chain = self.prompt | structured_chat

        self.initial_time = time.time()
        response: AIMessage = self.chain.invoke({"prompt": "Tell me a joke about cats"})
        self.final_time = time.time()

        self.get_logger().info(f"Prompt: Tell me a joke about cats")
        self.get_logger().info(f"Response: {response}")
        self.get_logger().info(
            f"Time elapsed: {self.final_time - self.initial_time:.2f} seconds"
        )
        # self.get_logger().info(
        #     f"Tokens per second: {response.usage_metadata['output_tokens'] / (self.final_time - self.initial_time):.2f} t/s"
        # )


def main():
    rclpy.init()
    node = ChatLlamaStructuredDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
