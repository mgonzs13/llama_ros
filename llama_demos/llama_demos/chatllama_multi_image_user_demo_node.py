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
import rclpy
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_ros.langchain import ChatLlamaROS


def main():
    tokens = 0
    initial_time = -1
    eval_time = -1

    rclpy.init()
    chat = ChatLlamaROS(temp=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are an IA that answer questions."),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {
                        "type": "text",
                        "text": (
                            "<__media__><__media__>\n"
                            "Who is the character in the middle of this first image and what type of food is the girl holding in this second image?"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": "https://pics.filmaffinity.com/Dragon_Ball_Bola_de_Dragaon_Serie_de_TV-973171538-large.jpg",
                    },
                    {
                        "type": "image_url",
                        "image_url": "https://i.pinimg.com/474x/32/89/17/328917cc4fe3bd4cfbe2d32aa9cc6e98.jpg",
                    },
                ]
            ),
        ]
    )

    chain = prompt | chat | StrOutputParser()

    initial_time = time.time()
    for text in chain.stream({}):
        tokens += 1
        print(text, end="", flush=True)
        if eval_time < 0:
            eval_time = time.time()

    print("", end="\n", flush=True)

    end_time = time.time()
    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
