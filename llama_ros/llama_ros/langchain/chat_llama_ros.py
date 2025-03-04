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

import cv2
import json
import uuid
import jinja2
import base64
import numpy as np
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Dict,
    Iterator,
    Sequence,
    Type,
    Union,
    Tuple,
)
from operator import itemgetter

from pydantic import BaseModel, model_validator
from jinja2.sandbox import ImmutableSandboxedEnvironment
from langchain_core.output_parsers import (
    PydanticToolsParser,
    JsonOutputKeyToolsParser,
    PydanticOutputParser,
    JsonOutputParser,
)

from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    InvalidToolCall,
    FunctionMessage,
    ToolCall,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import (
    _create_usage_metadata,
    _lc_invalid_tool_call_to_openai_tool_call,
    _lc_tool_call_to_openai_tool_call,
    _format_message_content,
    _convert_dict_to_message,
)

from action_msgs.msg import GoalStatus
from llama_ros.langchain import LlamaROSCommon
from llama_msgs.msg import ChatMessage, Content
from llama_msgs.srv import Detokenize
from llama_msgs.action import GenerateChatCompletions
import openai


class ChatLlamaROS(BaseChatModel, LlamaROSCommon):
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        message_dict: Dict[str, Any] = {
            "content": _format_message_content(message.content)
        }
        if (name := message.name or message.additional_kwargs.get("name")) is not None:
            message_dict["name"] = name

        # populate role and additional message data
        if isinstance(message, ChatMessage):
            message_dict["role"] = message.role
        elif isinstance(message, HumanMessage):
            message_dict["role"] = "user"
        elif isinstance(message, AIMessage):
            message_dict["role"] = "assistant"
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs[
                    "function_call"
                ]
            if message.tool_calls or message.invalid_tool_calls:
                message_dict["tool_calls"] = [
                    _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
                ] + [
                    _lc_invalid_tool_call_to_openai_tool_call(tc)
                    for tc in message.invalid_tool_calls
                ]
            elif "tool_calls" in message.additional_kwargs:
                message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
                tool_call_supported_props = {"id", "type", "function"}
                message_dict["tool_calls"] = [
                    {
                        k: v
                        for k, v in tool_call.items()
                        if k in tool_call_supported_props
                    }
                    for tool_call in message_dict["tool_calls"]
                ]
            else:
                pass
            # If tool calls present, content null value should be None not empty string.
            if "function_call" in message_dict or "tool_calls" in message_dict:
                message_dict["content"] = message_dict["content"] or None

        elif isinstance(message, SystemMessage):
            message_dict["role"] = message.additional_kwargs.get(
                "__openai_role__", "system"
            )
        elif isinstance(message, FunctionMessage):
            message_dict["role"] = "function"
        elif isinstance(message, ToolMessage):
            message_dict["role"] = "tool"
            message_dict["tool_call_id"] = message.tool_call_id

            supported_props = {"content", "role", "tool_call_id"}
            message_dict = {
                k: v for k, v in message_dict.items() if k in supported_props
            }
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop

        return {
            "messages": [self._convert_message_to_dict(m) for m in messages],
            **self._default_params,
            **kwargs,
        }

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        generations = []

        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

        token_usage = response_dict.get("usage")
        for res in response_dict["choices"]:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            generation_info = generation_info or {}
            generation_info["finish_reason"] = (
                res.get("finish_reason")
                if res.get("finish_reason") is not None
                else generation_info.get("finish_reason")
            )
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response_dict.get("model", self.model_name),
            "system_fingerprint": response_dict.get("system_fingerprint", ""),
        }

        if isinstance(response, openai.BaseModel) and getattr(
            response, "choices", None
        ):
            message = response.choices[0].message  # type: ignore[attr-defined]
            if hasattr(message, "parsed"):
                generations[0].message.additional_kwargs["parsed"] = message.parsed
            if hasattr(message, "refusal"):
                generations[0].message.additional_kwargs["refusal"] = message.refusal

        return ChatResult(generations=generations, llm_output=llm_output)

    def _send_llama_request(self, payload: Dict[str, Any]) -> Any:
        chat_request = GenerateChatCompletions.Goal()
        chat_request.messages = []
        
        print(payload)

        for message in payload["messages"]:
            print(f'message: {message}')
            chat_message = ChatMessage()
            chat_message.role = message["role"]
            if type(message["content"]) == str:
                chat_message.content = message["content"]
            elif type(message["content"]) == list:
                for content in message["content"]:
                    chat_content = Content()
                    chat_content.text = content["text"]
                    chat_content.type = content["type"]
                    chat_message.content_parts.append(chat_content)
            chat_request.messages.append(chat_message)
        
        sampling_config = self.llama_client._create_sampling_config("")
        chat_request.sampling_config = sampling_config

        result, status = self.llama_client.generate_chat_completions(chat_request)
        response = result.response

        print(response)
        return {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        response = self._send_llama_request(payload)
        return self._create_chat_result(response, generation_info)
