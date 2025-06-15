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


import sys
import time
import rclpy
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import AIMessage


def main():
    if len(sys.argv) < 2:
        prompt = "Here we have a book, a laptop and a nail. Please tell me how to stack them onto each other in a stable manner in English."
    else:
        prompt = " ".join(sys.argv[1:])

    rclpy.init()
    initial_time = -1
    chat = ChatLlamaROS(temp=0.2, penalty_last_n=8, enable_thinking=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": f"{prompt}"},
                ]
            ),
        ]
    )
    chain = prompt | chat

    initial_time = time.time()
    response: AIMessage = chain.invoke({})
    final_time = time.time()

    print(f"Prompt: {prompt}")
    print(f"Response: {response.content.strip()}")

    if "reasoning_content" in response.additional_kwargs:
        print(
            f"Reasoning length: {len(response.additional_kwargs['reasoning_content'])} characters"
        )
    else:
        print("No reasoning content. Are you sure you are using a reasoning model?")

    print(f"Time elapsed: {final_time - initial_time:.2f} seconds")
    print(
        f"Tokens per second: {response.usage_metadata['output_tokens'] / (final_time - initial_time):.2f} t/s"
    )
    rclpy.shutdown()


if __name__ == "__main__":
    main()
