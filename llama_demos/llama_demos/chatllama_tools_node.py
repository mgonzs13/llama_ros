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
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from llama_ros.langchain import ChatLlamaROS
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from random import randint

from pydantic import BaseModel, Field


class City(BaseModel):
    city: str = Field(..., description="City to get the temperature")
    # inhabitants: int = Field(..., description="Number of inhabitants")

@tool
def get_current_temperature(wheater_input: City) -> int:
    """Get the current temperature of a city"""
    return randint(-10, 40)

@tool
def get_max_temperature(wheater_input: City) -> int:
    """Get the max temperature of a city"""
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
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"type": "text", "text": "{prompt}"},
                    ]
                )
            ]
        )
                
        llm_temperature = self.chat.with_structured_output(get_max_temperature, method="function_calling")
        llm_city = self.chat.with_structured_output(City, method="json_schema")
        
        temperature_chain = self.prompt | llm_temperature
        city_chain = self.prompt | llm_city
        
        city_res = city_chain.invoke({"prompt": "What is the capital of Spain? And how many inhabitants does it have?"})
        print(city_res)
        
        temperature_res = temperature_chain.invoke({"prompt": "What is the temperature in Madrid?"})
        print(temperature_res)



def main():
    rclpy.init()
    node = ChatLlamaToolsDemoNode()
    node.send_prompt()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
