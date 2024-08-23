from typing import Any, List, Optional, Union, Dict
from pydantic import root_validator

from llama_msgs.msg import Message
from llama_msgs.srv import ChatMessages
from llama_ros.llama_client_node import LlamaClientNode
from llama_ros.langchain import LlamaROS

from langchain_core.language_models.base import PromptValue
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import Callbacks


class FormatttedChatLlamaROS(LlamaROS):
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["llama_client"] = LlamaClientNode.get_instance(
            values["namespace"])
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "formattedllamaros"

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = []
        for p in prompts:
            prompt_msgs = ChatMessages.Request()
            if type(p) == ChatPromptValue:
                for msg in p.messages:
                    prompt_msgs.messages.append(Message(role=msg.type, content=msg.content))

                prompt_str = self.llama_client.get_instance().format_chat_prompt(prompt_msgs)
            else:
                prompt_msg = Message(role='user', content=p.to_string())
                prompt_srv_msg = ChatMessages.Request(messages=[prompt_msg])
                prompt_str = self.llama_client.get_instance().format_chat_prompt(prompt_srv_msg)
            
            prompt_strings.append(prompt_str.formatted_prompt)

        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)