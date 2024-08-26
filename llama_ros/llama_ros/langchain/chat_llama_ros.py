from typing import Any, List, Optional, Dict, Iterator
from pydantic import root_validator
import urllib.request
import numpy as np
from cv_bridge import CvBridge
import cv2

from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.msg import Message
from llama_msgs.srv import ChatMessages
from llama_msgs.action import GenerateResponse
from llama_msgs.msg import LogitBias

from langchain.callbacks.manager import CallbackManagerForLLMRun
from action_msgs.msg import GoalStatus
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ChatLlamaROS(BaseChatModel):
    
    namespace: str = "llama"
    llama_client: LlamaClientNode = None
    cv_bridge: CvBridge = CvBridge()

    # sampling params
    n_prev: int = 64
    n_probs: int = 1
    min_keep: int = 0

    ignore_eos: bool = False
    logit_bias: Dict[int, float] = {}

    temp: float = 0.80
    dynatemp_range: float = 0.0
    dynatemp_exponent: float = 1.0

    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    tfs_z: float = 1.00
    typical_p: float = 1.00

    penalty_last_n: int = 64
    penalty_repeat: float = 1.00
    penalty_freq: float = 0.00
    penalty_present: float = 0.00

    mirostat: int = 0
    mirostat_eta: float = 0.10
    mirostat_tau: float = 5.0

    penalize_nl: bool = False

    samplers_sequence: str = "kfypmt"

    grammar: str = ""
    grammar_schema: str = ""

    penalty_prompt_tokens: List[int] = []
    use_penalty_prompt_tokens: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["llama_client"] = LlamaClientNode.get_instance(
            values["namespace"])
        return values

    def _create_action_goal(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        image_url: Optional[str] = None,
        image: Optional[np.ndarray] = None,
    ) -> GenerateResponse.Result:

        goal = GenerateResponse.Goal()
        goal.prompt = prompt
        goal.reset = True

        # load image
        if image_url or image:

            if image_url and not image:
                req = urllib.request.Request(
                    image_url, headers={"User-Agent": "Mozilla/5.0"})
                response = urllib.request.urlopen(req)
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)

            goal.image = self.cv_bridge.cv2_to_imgmsg(image)

        # add stop
        if stop:
            goal.stop = stop

        # sampling params
        goal.sampling_config.n_prev = self.n_prev
        goal.sampling_config.n_probs = self.n_probs
        goal.sampling_config.min_keep = self.min_keep

        goal.sampling_config.ignore_eos = self.ignore_eos
        for key in self.logit_bias:
            lb = LogitBias()
            lb.token = key
            lb.bias = self.logit_bias[key]
            goal.sampling_config.logit_bias.data.append(lb)

        goal.sampling_config.temp = self.temp
        goal.sampling_config.dynatemp_range = self.dynatemp_range
        goal.sampling_config.dynatemp_exponent = self.dynatemp_exponent

        goal.sampling_config.top_k = self.top_k
        goal.sampling_config.top_p = self.top_p
        goal.sampling_config.min_p = self.min_p
        goal.sampling_config.tfs_z = self.tfs_z
        goal.sampling_config.typical_p = self.typical_p

        goal.sampling_config.penalty_last_n = self.penalty_last_n
        goal.sampling_config.penalty_repeat = self.penalty_repeat
        goal.sampling_config.penalty_freq = self.penalty_freq
        goal.sampling_config.penalty_present = self.penalty_present

        goal.sampling_config.mirostat = self.mirostat
        goal.sampling_config.mirostat_eta = self.mirostat_eta
        goal.sampling_config.mirostat_tau = self.mirostat_tau

        goal.sampling_config.penalize_nl = self.penalize_nl

        goal.sampling_config.samplers_sequence = self.samplers_sequence

        goal.sampling_config.grammar = self.grammar
        goal.sampling_config.grammar_schema = self.grammar_schema

        goal.sampling_config.penalty_prompt_tokens = self.penalty_prompt_tokens
        goal.sampling_config.use_penalty_prompt_tokens = self.use_penalty_prompt_tokens

        return goal

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "chatllamaros"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,

    ) -> str:
        chat_messages = ChatMessages.Request()

        image = None
        image_url = None

        llama_client = self.llama_client.get_instance(self.namespace)

        for message in messages:
            if type(message.content) == str:
                chat_messages.messages.append(Message(role=message.type, content=message.content))
            else:
                for single_content in message.content:
                    if single_content['type'] == 'text':
                        chat_messages.messages.append(Message(role=message.type, content=single_content['text']))
                    elif single_content['type'] == 'image_url':
                        image_url = single_content['image_url']['url']
                    else:
                        chat_messages.messages.append(Message(role=message.type, content=str(single_content)))
        
        formatted_prompt = llama_client.format_chat_prompt(chat_messages).formatted_prompt

        goal_action = self._create_action_goal(formatted_prompt, stop, image_url, image, **kwargs)

        result, status = LlamaClientNode.get_instance(
            self.namespace).generate_response(goal_action)
        
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
        chat_messages = ChatMessages.Request()

        image = None
        image_url = None

        llama_client = self.llama_client.get_instance(self.namespace)

        for message in messages:
            if type(message.content) == str:
                chat_messages.messages.append(Message(role=message.type, content=message.content))
            else:
                for single_content in message.content:
                    if single_content['type'] == 'text':
                        chat_messages.messages.append(Message(role=message.type, content=single_content['text']))
                    elif single_content['type'] == 'image_url':
                        image_url = single_content['image_url']['url']
                    else:
                        chat_messages.messages.append(Message(role=message.type, content=str(single_content)))
        
        formatted_prompt = llama_client.format_chat_prompt(chat_messages).formatted_prompt

        goal_action = self._create_action_goal(formatted_prompt, stop, image_url, image **kwargs)

        for pt in LlamaClientNode.get_instance(
                self.namespace).generate_response(goal_action, stream=True):

            if run_manager:
                run_manager.on_llm_new_token(pt.text, verbose=self.verbose,)

            yield ChatGenerationChunk(message=AIMessageChunk(content=pt.text))