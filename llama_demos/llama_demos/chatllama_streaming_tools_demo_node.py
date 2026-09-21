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
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from llama_ros.langchain import ChatLlamaROS
import asyncio


@tool
def get_inhabitants(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(4_000_000, 8_000_000)


@tool
def get_curr_temperature(city: str) -> int:
    """Get the current temperature of a city"""
    return randint(20, 30)


async def main():
    rclpy.init()
    chat = ChatLlamaROS(temp=0.0, penalty_repeat=1.3, penalty_freq=1.3)

    messages = [
        HumanMessage("What is the current temperature in Madrid? And its inhabitants?")
    ]

    print(f"\nPrompt: {messages[0].content}")
    llm_tools = chat.bind_tools(
        [get_inhabitants, get_curr_temperature], tool_choice="any"
    )

    initial_time = time.time()

    # stream the tool-call turn, printing each tool call as it completes
    first = True
    gathered = None
    tools_output_tokens = 0
    async for chunk in llm_tools.astream(messages):
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk

        if (
            chunk.tool_call_chunks
            and chunk.tool_call_chunks[-1]["args"]
            and "}" in chunk.tool_call_chunks[-1]["args"]
        ):
            print(
                f"Tool received: {gathered.tool_calls[-1]['name']}({gathered.tool_calls[-1]['args']})"
            )

        if chunk.usage_metadata:
            tools_output_tokens = chunk.usage_metadata["output_tokens"]

    tools_end_time = time.time()

    # execute the tools and append their results to the conversation
    messages.append(gathered)
    for tool_call in gathered.tool_calls:
        selected_tool = {
            "get_inhabitants": get_inhabitants,
            "get_curr_temperature": get_curr_temperature,
        }[tool_call["name"]]

        tool_msg = selected_tool.invoke(tool_call)
        formatted_output = (
            f"{tool_call['name']}"
            f"({', '.join(f'{k}={v}' for k, v in tool_call['args'].items())})"
            f" = {tool_msg.content}"
        )
        print(f"Calling tool: {formatted_output}")

        tool_msg.additional_kwargs = {"args": tool_call["args"]}
        messages.append(tool_msg)

    # stream the final response generated with the tool results
    print("\nResponse: ", end="", flush=True)
    response_output_tokens = 0
    async for chunk in chat.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

        if chunk.usage_metadata:
            response_output_tokens = chunk.usage_metadata["output_tokens"]
    print("", flush=True)

    end_time = time.time()

    tools_time = tools_end_time - initial_time
    response_time = end_time - tools_end_time

    print(f"\nTime to generate tools: {tools_time:.2f} s")
    print(f"Tokens per second (tools): {tools_output_tokens / tools_time:.2f} t/s")
    print(f"Time to generate last response: {response_time:.2f} s")
    print(
        f"Tokens per second (last response): {response_output_tokens / response_time:.2f} t/s"
    )

    rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
