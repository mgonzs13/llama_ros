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
from langchain_core.output_parsers import (
    PydanticToolsParser,
    JsonOutputKeyToolsParser,
    PydanticOutputParser,
    JsonOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.utils.pydantic import is_basemodel_subclass
import base64
import cv2
import numpy as np
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel
import uuid
from ament_index_python.packages import get_package_share_directory

from llama_ros.langchain import LlamaROSCommon
from llama_msgs.msg import Message
from action_msgs.msg import GoalStatus
from llama_msgs.srv import Detokenize
from llama_msgs.srv import FormatChatMessages
from llama_msgs.action import GenerateResponse
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
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

DEFAULT_TEMPLATE = """{% if tools_grammar %}
    {{- '<|im_start|>assistant\n' }}
    {{- 'You are an assistant. You output in JSON format. The key "tool_calls" is a list of possible tools. For each tool, the format is {name, arguments}. You can use the following tools:' }}
    {% for tool in tools_grammar %}
        {% if not loop.last %}
            {{- tool }}
        {% else %}
            {{- tool + '<|im_end|>' }}
        {% endif %}
    {% endfor %}
{% endif %}

{% for message in messages %}
    {% if (loop.last and add_generation_prompt) or not loop.last %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {% else %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
    {{- '<|im_start|>assistant' }}
{% endif %}
"""


class ChatLlamaROS(BaseChatModel, LlamaROSCommon):

    use_llama_template: bool = False

    use_gguf_template: bool = True

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

    def _json_schema_to_definition(self, input_json):
        # Extract the tool name
        tool_name = input_json["properties"]["name"]["const"]

        # Extract and map arguments to desired format
        properties = input_json["properties"]["arguments"]["properties"]
        transformed_properties = {arg: prop["type"] for (arg, prop) in properties.items()}

        # Create the transformed object
        return {"name": tool_name, "arguments": transformed_properties}

    def _generate_prompt(self, messages: List[dict[str, str]], **kwargs) -> str:
        tools_grammar = kwargs.get("tools_grammar", None)

        if self.use_llama_template:
            chat_template = DEFAULT_TEMPLATE
        else:
            chat_template = self.model_metadata.tokenizer.chat_template

        formatted_tools = []
        if tools_grammar:
            list_options = json.loads(tools_grammar)["properties"]["tool_calls"]["items"]
            for key in list_options.keys():
                if key.endswith("Of"):
                    list_key = key

            formatted_tools = [
                self._json_schema_to_definition(tool) for tool in list_options[list_key]
            ]
            formatted_tools = [
                {key: tool[key] for key in sorted(tool, reverse=True)}
                for tool in formatted_tools
            ]

        bos_token = self.llama_client.detokenize(
            Detokenize.Request(tokens=[self.model_metadata.tokenizer.bos_token_id])
        ).text

        if self.use_gguf_template or self.use_llama_template:
            formatted_prompt = self.jinja_env.from_string(chat_template).render(
                messages=messages,
                add_generation_prompt=True,
                bos_token=bos_token,
                tools_grammar=[json.dumps(tool) for tool in formatted_tools],
            )
            return formatted_prompt
        else:
            ros_messages = [
                Message(content=message["content"], role=message["role"])
                for message in messages
            ]
            return self.llama_client.format_chat_prompt(
                FormatChatMessages.Request(messages=ros_messages)
            ).formatted_prompt

    def _convert_content(
        self, content: Union[Dict[str, str], str, List[str], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
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
            converted_msg = [
                {"role": "user", "content": self._convert_content(message.content)}
            ]
            return converted_msg

        elif isinstance(message, AIMessage):
            all_messages = []

            contents = self._convert_content(message.content)
            if isinstance(contents, dict):
                contents = [contents]
            contents = [
                content
                for content in contents
                if content["type"] == "text" and content["text"] != ""
            ]

            all_messages.extend(
                [{"role": "assistant", "content": content} for content in contents]
            )

            return all_messages

        elif isinstance(message, SystemMessage):
            converted_msg = [
                {"role": "system", "content": self._convert_content(message.content)}
            ]
            return converted_msg

        elif isinstance(message, ToolMessage):
            tool_args = message.additional_kwargs.get("args", {})
            formatted_args = ", ".join([f"{value}" for _, value in tool_args.items()])
            formatted_content = f"{message.name}({formatted_args}): {message.content}"
            return [
                {
                    "role": "tool",
                    "content": {"type": "text", "text": formatted_content},
                    "tool_call_id": message.tool_call_id,
                }
            ]
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    def _extract_data_from_messages(
        self, messages: List[BaseMessage]
    ) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
        new_messages = []
        image_url = None
        image = None

        def process_content(role, content):
            nonlocal image, image_url
            if isinstance(content, str):
                new_messages.append({"role": role, "content": content})
            elif isinstance(content, dict):
                if content["type"] == "text":
                    new_messages.append({"role": role, "content": content["text"]})
                elif content["type"] == "image":
                    image = content["image"]
                elif content["type"] == "image_url":
                    image_url = content["image_url"]

        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                for single_content in content:
                    process_content(role, single_content)
            else:
                process_content(role, content)

        return new_messages, image_url, image

    def _create_chat_generations(
        self, response: GenerateResponse.Result, method: str
    ) -> List[BaseMessage]:
        chat_gen = None

        if method == "function_calling":
            ai_message = AIMessage(content="", tool_calls=[])
            parsed_output = json.loads(response.text)
            for tool in parsed_output["tool_calls"]:
                ai_message.tool_calls.append(
                    {
                        "name": tool["name"],
                        "args": tool["arguments"],
                        "type": "tool_call",
                        "id": f'{tool["name"]}_{uuid.uuid4()}',
                    }
                )

            chat_gen = ChatGeneration(message=ai_message)
        else:
            chat_gen = ChatGeneration(message=AIMessage(content=response.text))

        return ChatResult(generations=[chat_gen])

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

        formatted_prompt = self._generate_prompt(chat_messages, **kwargs)

        goal_action = self._create_action_goal(
            formatted_prompt, stop, image_url, image, **kwargs
        )

        result, status = self.llama_client.generate_response(goal_action)
        response = result.response

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""

        return self._create_chat_generations(response, kwargs.get("method", "chat"))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if kwargs.get("method") == "function_calling":
            raise ValueError(
                "Streaming is not supported when using 'function_calling' method."
            )

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
        method: Literal[
            "function_calling", "json_schema", "json_mode"
        ] = "function_calling",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", "auto" choices like OpenAI
            tool-calling API. should be a dict of the form to force this tool
            {"type": "function", "function": {"name": <<tool_name>>}}.
        """

        formatted_tools = []

        for tool in tools:
            formatted_tools.append(convert_to_openai_tool(tool)["function"])

        tool_names = [ft["name"] for ft in formatted_tools]
        valid_choices = ["all", "one", "any"]

        is_valid_choice = tool_choice in valid_choices

        chosen_tool = [f for f in formatted_tools if f["name"] == tool_choice]

        if not chosen_tool and not is_valid_choice:
            raise ValueError(
                f"Tool choice {tool_choice=} was specified, but the only "
                f"provided tools were {tool_names}."
            )

        grammar = {}

        if method == "json_mode" or method == "json_schema":
            grammar = chosen_tool[0]["parameters"]
        else:
            grammar = {
                "type": "object",
                "properties": {
                    "tool_calls": {
                        "type": "array",
                        "items": {"type": "object"},
                        "maxItems": 10,
                    }
                },
                "required": ["tool_calls"],
            }

            if chosen_tool:
                grammar["properties"]["tool_calls"]["items"]["oneOf"] = []
                new_action = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "const": chosen_tool[0]["name"]},
                        "arguments": chosen_tool[0]["parameters"],
                    },
                    "required": ["name", "arguments"],
                }
                grammar["properties"]["tool_calls"]["items"]["oneOf"].append(new_action)
            else:
                grammar["properties"]["tool_calls"]["items"][f"{tool_choice}Of"] = []
                for tool in formatted_tools:
                    new_action = {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "const": tool["name"]},
                            "arguments": tool["parameters"],
                        },
                        "required": ["name", "arguments"],
                    }
                    grammar["properties"]["tool_calls"]["items"][
                        f"{tool_choice}Of"
                    ].append(new_action)

        return super().bind(tools_grammar=json.dumps(grammar), method=method, **kwargs)

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

        if method == "json_mode" or method == "json_schema":
            tool_name = schema.__name__

            llm = self.bind_tools([schema], tool_choice=tool_name, method=method)
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
            schema = convert_to_openai_tool(schema)["function"]
            tool_name = schema["name"]

            llm = self.bind_tools([schema], tool_choice=tool_name, method=method)
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
