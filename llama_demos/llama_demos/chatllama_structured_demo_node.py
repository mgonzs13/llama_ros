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
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage
from llama_ros.langchain import ChatLlamaROS


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


def main():
    rclpy.init()
    chat = ChatLlamaROS(temp=0.2, penalty_last_n=8)

    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(
                template=[
                    {"type": "text", "text": "{prompt}"},
                ]
            ),
        ]
    )

    chain = prompt | chat.with_structured_output(Joke, method="function_calling")
    initial_time = time.time()
    response: AIMessage = chain.invoke({"prompt": "Tell me a joke about cats"})
    message: AIMessage = response["raw"]
    joke: Joke = response["parsed"]

    final_time = time.time()

    print(f"Prompt: Tell me a joke about cats")
    print(f"Response: {joke.model_dump_json()}")
    print(f"Time elapsed: {final_time - initial_time:.2f} seconds")
    print(
        f"Tokens per second: {message.usage_metadata['output_tokens'] / (final_time - initial_time):.2f} t/s"
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()
