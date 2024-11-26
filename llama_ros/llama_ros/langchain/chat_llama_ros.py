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

from typing import Any, Callable, List, Literal, Optional, Dict, Iterator, Sequence, Type, Union, Tuple, cast
from operator import itemgetter
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputKeyToolsParser, PydanticOutputParser, JsonOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.utils.pydantic import is_basemodel_subclass
import base64
import cv2
import numpy as np
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel
from pydantic import create_model

from llama_ros.langchain import LlamaROSCommon
from llama_msgs.msg import Message
from llama_msgs.srv import FormatChatMessages, Detokenize
from action_msgs.msg import GoalStatus
from langchain_core.utils.function_calling import convert_to_openai_tool
import json

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
        
        bos_token = self.llama_client.detokenize(Detokenize.Request(tokens=[self.model_metadata.tokenizer.bos_token_id])).text
        
        if chat_template:
            formatted_prompt = self.jinja_env.from_string(
                chat_template
            ).render(
                messages=messages,
                add_generation_prompt=True,
                bos_token=bos_token,
            )
            return formatted_prompt
        else:
            ros_messages = [Message(content=message["content"], role=message["role"]) for message in messages]
            return self.llama_client.format_chat_messages(ros_messages).formatted_prompt

    def _convert_content(self, content: Union[Dict[str, str], str, List[str], List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(content, str):
            return {"type": "text", "text": content}
        if isinstance(content, list) and len(content) == 1:
            return self._convert_content(content[0])
        elif isinstance(content, list):
            return [self._convert_content(c) for c in content]
        elif isinstance(content, dict):
            if content["type"] == "text":
                return {"type": "text", "text": content["text"]}
            elif content["type"] == "image_url":
                image_text = content["image_url"]["url"]
                if "data:image" in image_text:
                    image_data = image_text.split(",")[-1]
                    decoded_image = base64.b64decode(image_data)
                    np_image = np.frombuffer(decoded_image, np.uint8)
                    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                    
                    return {"type": "image", "image": image}
                else:
                    image_url = image_text
                    return {"type": "image_url", "image_url": image_url}


    def _convert_message_to_dict(self, message: BaseMessage) -> list[dict[str, str]]:
        if isinstance(message, HumanMessage):
            return [{"role": "user", "content": self._convert_content(message.content)}]
        
        elif isinstance(message, AIMessage):
            all_messages = []
            
            # Text messages
            all_messages.extend([{"role": "assistant", "content": content} for content in self._convert_content(message.content)])
            
            # Tool messages
            all_messages.extend([{"role": "assistant", "content": "", "tool_call_id": tc['id']} for tc in message.additional_kwargs["tool_calls"]])
            
            return all_messages
            
        elif isinstance(message, SystemMessage):
            return [{"role": "system", "content": self._convert_content(message.content)}]
        elif isinstance(message, ToolMessage):
            return [{"role": "tool", "content": self._convert_content(message.content), "tool_call_id": message.tool_call_id}]
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def _extract_data_from_messages(self, messages: List[BaseMessage]) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
        new_messages = []
        image_url = None
        image = None
                
        for message in messages:
            if message['content']['type'] == 'image':
                image = message['content']['image']
            elif message['content']['type'] == 'image_url':
                image_url = message['content']['image_url']
            else:
                new_messages.append({"role": message['role'], "content": message['content']['text']})
        
        return new_messages, image_url, image

    def _convert_completion_to_chat_function(
        tool_name: str,
        completion_or_chunks: Any,
        stream: bool,
    ):
        if not stream:
            completion = completion_or_chunks  # type: ignore
            tool_id = "call_" + "_0_" + tool_name + "_" + completion["id"]
            # TODO: Fix for legacy function calls
            chat_completion = {
                "id": "chat" + completion["id"],
                "object": "chat.completion",
                "created": completion["created"],
                "model": completion["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": tool_name,
                                "arguments": completion["choices"][0]["text"],
                            },
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": completion["choices"][0]["text"],
                                    },
                                }
                            ],
                        },
                        "logprobs": completion["choices"][0]["logprobs"],
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": completion["usage"],
            }
            return chat_completion


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        dict_messages = []
        for message in messages:
            dict_messages.extend(self._convert_message_to_dict(message))
        
        chat_messages, image_url, image = self._extract_data_from_messages(dict_messages)
        
        formatted_prompt = self._generate_prompt(chat_messages)

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        # TODO: Hay que adaptar la salida del modelo al formato estándar de OpenAI
        result, status = self.llama_client.generate_response(goal_action)

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""

        # TODO: Habría que adaptar esto para funciones
        generation = ChatGeneration(message=AIMessage(content=result.response.text.strip()))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:        
        dict_messages = []
        for message in messages:
            dict_messages.extend(self._convert_message_to_dict(message))
            
        chat_messages, image_url, image = self._extract_data_from_messages(dict_messages)
        
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
            
            
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[Dict[str, Dict], bool, str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", "auto" choices like OpenAI
            tool-calling API. should be a dict of the form to force this tool
            {"type": "function", "function": {"name": <<tool_name>>}}.
        """
        
        formatted_tools = []
        
        for tool in tools:
            formatted_tools.append(convert_to_openai_tool(tool)['function'])
                        
        tool_names = [ft['name'] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                valid_choices = ["all", "one", "any"]
                
                is_valid_choice = tool_choice in valid_choices
                
                chosen_tool = [
                    f for f in formatted_tools if f["name"] == tool_choice
                ]
                
                if not chosen_tool and not is_valid_choice:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
                    
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like 
                    this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )

        # kwargs["tool_choice"] = tool_choice
        # formatted_tools = [tool.model_json_schema() for tool in tools]
        return super().bind(tools_grammar=json.dumps(tool_choice['parameters']), **kwargs)


    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel], Type]] = None,
        *,
        include_raw: bool = False,
        method: Literal[
            "function_calling", "json_schema", "json_mode"
        ] = "function_calling",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        
        # if is_pydantic_schema is False:
        #     raise ValueError(
        #         "Schema must be a Pydantic model class or a dict of the form "
        #     )
        
        if method == "json_mode" or method == "json_schema":
            tool_name = schema.__name__
            
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            schema = convert_to_openai_tool(schema)['function']
            tool_name = schema["name"]  
            
            print(f"tool_name: {tool_name}")
                        
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name
            )   
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
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
