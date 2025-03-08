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
    TypeVar,
    Type,
    Union,
    Tuple,
    cast,
)
from operator import itemgetter
from functools import partial
import warnings
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
from langchain_core.runnables import Runnable, RunnableLambda
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
    _convert_to_openai_response_format,
    _oai_structured_outputs_parser,
    _is_pydantic_class,
)

from action_msgs.msg import GoalStatus
from llama_ros.langchain import LlamaROSCommon
from llama_msgs.msg import ChatMessage, Content, ChatReqTool, ChatTool, ChatToolCall
from llama_msgs.srv import Detokenize
from llama_msgs.action import GenerateChatCompletions
import openai
import json
from pydantic import Field

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]

class ChatLlamaROS(BaseChatModel, LlamaROSCommon):
    image_data: Optional[str] = Field(default=None, exclude=True)
    image_url: Optional[str] = Field(default=None, exclude=True)
    disabled_params: Optional[Dict[str, Any]] = Field(default=None)
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"
    
    def _extract_image_data(self, contents: Union[List[Dict[str, str]], str, Dict[str, str]]) -> Tuple[str, str]:
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
    
    def _parse_tool_choice(self, tool_choice: str) -> dict:
        if tool_choice == "auto":
            return ChatTool.TOOL_CHOICE_AUTO
        elif tool_choice == "required":
            return ChatTool.TOOL_CHOICE_REQUIRED
        else:
            return ChatTool.TOOL_CHOICE_NONE

    def _send_llama_request(self, payload: Dict[str, Any], **kwargs) -> Any:
        chat_request = GenerateChatCompletions.Goal()
        chat_request.add_generation_prompt = True
        chat_request.use_jinja = True
        chat_request.messages = []
        chat_request.tools = []
        chat_request.parallel_tool_calls = kwargs.get("parallel_tool_calls", True)

        if (self.image_url or self.image_data) is not None:
            chat_request.image = self._get_image(self.image_url, self.image_data)
            
        # TODO: get tool_choice from payload or kwargs
        chat_request.tool_choice = ChatTool.TOOL_CHOICE_AUTO

        for message in payload.get("messages", []):
            chat_message = ChatMessage()
            chat_message.role = message["role"]
            
            for tool_call in message.get("tool_calls", []):
                chat_tool_call = ChatToolCall()
                chat_tool_call.id = tool_call["id"]
                chat_tool_call.name = tool_call["function"]["name"]
                chat_tool_call.arguments = json.dumps(tool_call["function"]["arguments"])
                chat_message.tool_calls.append(chat_tool_call)

            if type(message["content"]) == str:
                chat_message.content = message["content"]
            elif type(message["content"]) == list:
                for content in message["content"]:
                    chat_content = Content()
                    chat_content.type = content["type"]
                    chat_content.text = content[content["type"]]
                    chat_message.content_parts.append(chat_content)
            chat_request.messages.append(chat_message)
            
        for tool in payload.get("tools", []):
            chat_req_tool = ChatReqTool()
            chat_req_tool.type = "function"
            chat_tool = ChatTool()
            chat_tool.name = tool["function"]["name"]
            chat_tool.description = tool["function"]["description"]
            chat_tool.parameters = json.dumps(tool["function"]["parameters"])
            chat_req_tool.function = chat_tool
            chat_request.tools.append(chat_req_tool)
        
        chat_request.sampling_config = self._set_sampling_config()
        
        result, _ = self.llama_client.generate_chat_completions(chat_request)
        
        return self._parse_chat_generation_response(result)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        generation_info = None
        response = self._send_llama_request(payload, **kwargs)
        result = self._create_chat_result(response, generation_info)
                
        return result
    
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
                tool_call_dict['id'] = tool_call.id
                tool_call_dict['type'] = 'function'
                tool_call_dict['function'] = {}
                tool_call_dict['function']['name'] = tool_call.name
                tool_call_dict['function']['arguments'] = tool_call.arguments

                msg_dict['tool_calls'].append(tool_call_dict)

            choice_dict['message'] = msg_dict
            result_dict['choices'].append(choice_dict)

        return result_dict
    
    def _filter_disabled_params(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.disabled_params:
            return kwargs
        filtered = {}
        for k, v in kwargs.items():
            # Skip param
            if k in self.disabled_params and (
                self.disabled_params[k] is None or v in self.disabled_params[k]
            ):
                continue
            # Keep param
            else:
                filtered[k] = v
        return filtered
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
    
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if strict is not None and method == "json_mode":
            raise ValueError(
                "Argument `strict` is not supported with `method`='json_mode'"
            )
        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "json_schema":
            # Check for Pydantic BaseModel V1
            if (
                is_pydantic_schema and issubclass(schema, BaseModelV1)  # type: ignore[arg-type]
            ):
                warnings.warn(
                    "Received a Pydantic BaseModel V1 schema. This is not supported by "
                    'method="json_schema". Please use method="function_calling" '
                    "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                    'Overriding to method="function_calling".'
                )
                method = "function_calling"

        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            bind_kwargs = self._filter_disabled_params(
                tool_choice=tool_name,
                parallel_tool_calls=False,
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )

            llm = self.bind_tools([schema], **bind_kwargs)
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            response_format = _convert_to_openai_response_format(schema, strict=strict)
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": convert_to_openai_tool(schema),
                },
            )
            if is_pydantic_schema:
                output_parser = RunnableLambda(
                    partial(_oai_structured_outputs_parser, schema=cast(type, schema))
                ).with_types(output_type=cast(type, schema))
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

