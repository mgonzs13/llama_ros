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
from random import randint
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from llama_ros.langchain import ChatLlamaROS


@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)


def main():
    rclpy.init()
    chat = ChatLlamaROS(temp=0.0)

    messages = [
        HumanMessage("What is the current temperature in Madrid? And its inhabitants?")
    ]

    print(f"\nPrompt: {messages[0].content}")
    llm_tools = chat.bind_tools(
        [get_inhabitants, get_curr_temperature], tool_choice="any"
    )

    initial_time = time.time()
    all_tools_res: AIMessage = llm_tools.invoke(messages)
    tools_time = time.time()

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
        print(f"Calling tool: {formatted_output}")

        tool_msg.additional_kwargs = {"args": tool["args"]}
        messages.append(tool_msg)

    res: AIMessage = llm_tools.invoke(messages)
    eval_time = time.time()
    print(f"\nResponse: {res.content}")

    time_generate_tools = tools_time - initial_time
    time_last_response = eval_time - tools_time
    print(f"Time to generate tools: {time_generate_tools:.2f} s")
    print(
        f"Tokens per second (tools): {all_tools_res.usage_metadata['output_tokens'] / time_generate_tools:.2f} t/s"
    )

    print(f"Time to generate last response: {time_last_response:.2f} s")
    print(
        f"Tokens per second (last response): {res.usage_metadata['output_tokens'] / time_last_response:.2f} t/s"
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()
