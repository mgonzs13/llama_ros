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

from typing import Any, List, Optional, Dict, Iterator
import base64
import cv2
import numpy as np
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from llama_ros.langchain import LlamaROSCommon
from llama_msgs.msg import Message
from llama_msgs.srv import FormatChatMessages
from action_msgs.msg import GoalStatus

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ChatLlamaROS(BaseChatModel, LlamaROSCommon):

    jinja_env: ImmutableSandboxedEnvironment = ImmutableSandboxedEnvironment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
    )
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"

    def _generate_prompt(self, messages: List[dict[str, str]]) -> str:        
        chat_template = self.model_metadata.tokenizer.chat_template
        
        if chat_template:
            formatted_prompt = self.jinja_env.from_string(
                chat_template
            ).render(
                messages=messages,
                add_generation_prompt=True,
            )
            return formatted_prompt
        else:
            ros_messages = [Message(content=message["content"], role=message["role"]) for message in messages]
            return self.llama_client.format_chat_messages(ros_messages).formatted_prompt

    def _messages_to_chat_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[FormatChatMessages.Request, Optional[str], Optional[np.ndarray]]:

        chat_messages = []
        image_url = None
        image = None

        for message in messages:
            role = "user" if message.type.lower() == "human" else message.type

            if isinstance(message.content, str):
                chat_messages.append({"role": role, "content": message.content})
            else:
                for single_content in message.content:
                    if isinstance(single_content, str):
                        chat_messages.append({"role": role, "content": single_content})
                    elif single_content["type"] == "text":
                        chat_messages.append(
                            {"role": role, "content": single_content["text"]}
                        )
                    elif single_content["type"] == "image_url":
                        image_text = single_content["image_url"]["url"]
                        if "data:image" in image_text:
                            image_data = image_text.split(",")[-1]
                            decoded_image = base64.b64decode(image_data)
                            np_image = np.frombuffer(decoded_image, np.uint8)
                            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                        else:
                            image_url = image_text

        return chat_messages, image_url, image

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        chat_messages, image_url, image = self._messages_to_chat_messages(messages)
        formatted_prompt = self._generate_prompt(chat_messages)

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        result, status = self.llama_client.generate_response(goal_action)

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""

        generation = ChatGeneration(message=AIMessage(content=result.response.text))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        chat_messages, image_url, image = self._messages_to_chat_messages(messages)
        formatted_prompt = self._generate_prompt(chat_messages)

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        for pt in self.llama_client.generate_response(goal_action, stream=True):

            if run_manager:
                run_manager.on_llm_new_token(
                    pt.text,
                    verbose=self.verbose,
                )

            yield ChatGenerationChunk(message=AIMessageChunk(content=pt.text))
