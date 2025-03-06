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
from pydantic import Field


class ChatLlamaROS(BaseChatModel, LlamaROSCommon):
    image_data: Optional[str] = Field(default=None, exclude=True)
    image_url: Optional[str] = Field(default=None, exclude=True)
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"
    
    def _extract_image_data(self, contents: Union[List[Dict[str, str]], str, Dict[str, str]]) -> Tuple[str, str]:
        image_data = None
        image_url = None

        if type(contents) == str:
            return contents

        if type(contents) == list:
            for content in contents:
                if content["type"] == "image_url":
                    self.image_url = content["image_url"]['url']
                    contents.remove(content)
                    return contents
                elif content["type"] == "image":
                    self.image_data = content["image"]
                    contents.remove(content)
                    return contents
        elif type(contents) == dict:
            if contents["type"] == "image_url":
                self.image_url = contents["image_url"]['url']
                return {"type": "text", "text": ""}

            elif contents["type"] == "image":
                self.image_data = contents["image"]
                return {"type": "text", "text": ""}     

        return contents

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        message_content = _format_message_content(message.content)
        content = self._extract_image_data(message_content)
                
        message_dict: Dict[str, Any] = {
            "content": content,
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
            "model_name": response_dict.get("model", response['model']),
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

    def _send_llama_request(self, payload: Dict[str, Any], **kwargs) -> Any:
        chat_request = GenerateChatCompletions.Goal()
        chat_request.add_generation_prompt = True
        chat_request.use_jinja = True
        chat_request.messages = []

        if (self.image_url or self.image_data) is not None:
            chat_request.image = self._get_image(self.image_url, self.image_data)

        for message in payload["messages"]:
            chat_message = ChatMessage()
            chat_message.role = message["role"]
            if type(message["content"]) == str:
                chat_message.content = message["content"]
            elif type(message["content"]) == list:
                for content in message["content"]:
                    chat_content = Content()
                    chat_content.type = content["type"]
                    chat_content.text = content[content["type"]]
                    chat_message.content_parts.append(chat_content)
            chat_request.messages.append(chat_message)
        
        chat_request.sampling_config = self._set_sampling_config()

        result, _ = self.llama_client.generate_chat_completions(chat_request)

        return self._parse_chat_generation_response(result)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.image_url = None
        self.image_data = None

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        response = self._send_llama_request(payload)
        return self._create_chat_result(response, generation_info)
    
    def _parse_chat_generation_response(self, result: GenerateChatCompletions.Result) -> dict:
        result_dict = {}

        result_dict['id'] = result.id
        result_dict['created'] = result.created
        result_dict['model'] = result.model
        result_dict['object'] = result.object
        result_dict['system_fingerprint'] = result.system_fingerprint
        result_dict['choices'] = []

        for choice in result.choices:
            choice_dict = {}
            choice_dict['finish_reason'] = choice.finish_reason
            choice_dict['index'] = choice.index

            msg_dict = {}
            msg_dict['content'] = choice.message.content
            msg_dict['role'] = choice.message.role
            msg_dict['reasoning_content'] = choice.message.reasoning_content
            msg_dict['tool_name'] = choice.message.tool_name
            msg_dict['tool_call_id'] = choice.message.tool_call_id
            msg_dict['content_parts'] = []
            msg_dict['tool_calls'] = []
            
            for content in choice.message.content_parts:
                content_dict = {}
                content_dict['type'] = content.type
                content_dict['text'] = content.text

                msg_dict['content_parts'].append(content_dict)
            for tool_call in choice.message.tool_calls:
                tool_call_dict = {}
                tool_call_dict['name'] = tool_call.name
                tool_call_dict['arguments'] = tool_call.arguments
                tool_call_dict['id'] = tool_call.id

                msg_dict['tool_calls'].append(tool_call_dict)

            choice_dict['message'] = msg_dict
            result_dict['choices'].append(choice_dict)

        return result_dict
