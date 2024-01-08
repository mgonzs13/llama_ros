# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Any, Dict, List, Optional
from pydantic import root_validator

from simple_node import Node
from simple_node.actions.action_client import ActionClient
from rclpy.client import Client

from llama_msgs.msg import LogitBias
from llama_msgs.action import GenerateResponse
from llama_msgs.srv import Tokenize

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class LlamaROS(LLM):

    node: Node

    action_name: str = "/llama/generate_response"
    action_client: ActionClient = None

    tokenize_srv_name: str = "/llama/tokenize"
    tokenize_srv_client: Client = None

    # sampling params
    n_prev: int = 64
    n_probs: int = 1

    ignore_eos: bool = False
    logit_bias: Dict[int, float] = {}

    temp: float = 0.80

    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    tfs_z: float = 1.00
    typical_p: float = 1.00

    penalty_last_n: int = 64
    penalty_repeat: float = 1.10
    penalty_freq: float = 0.00
    penalty_present: float = 0.00

    mirostat: int = 0
    mirostat_eta: float = 0.10
    mirostat_tau: float = 5.0

    penalize_nl: bool = True

    samplers_sequence: str = "kfypmt"
    grammar: str = ""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:

        node: Node = values["node"]

        action_name = values["action_name"]
        values["action_client"] = node.create_action_client(
            GenerateResponse, action_name)

        tokenize_srv_name = values["tokenize_srv_name"]
        values["tokenize_srv_client"] = node.create_client(
            Tokenize, tokenize_srv_name)

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {**{"action_name": self.action_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        return "llamaros"

    def cancel(self) -> None:
        self.action_client.cancel_goal()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        goal = GenerateResponse.Goal()
        goal.prompt = prompt
        goal.reset = True

        # sampling params
        goal.sampling_config.n_prev = self.n_prev
        goal.sampling_config.n_probs = self.n_probs

        goal.sampling_config.ignore_eos = self.ignore_eos
        for key in self.logit_bias:
            lb = LogitBias()
            lb.token = key
            lb.bias = self.logit_bias[key]
            goal.sampling_config.logit_bias.data.append(lb)

        goal.sampling_config.temp = self.temp

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

        # send goal
        self.action_client.wait_for_server()
        self.action_client.send_goal(goal)
        self.action_client.wait_for_result()
        result: GenerateResponse.Result = self.action_client.get_result()

        if self.action_client.is_canceled():
            return ""
        return result.response.text

    def get_num_tokens(self, text: str) -> int:

        req = Tokenize.Request()
        req.prompt = text

        self.tokenize_srv_client.wait_for_service()
        res = self.tokenize_srv_client.call(req)
        tokens = res.tokens

        return len(tokens)
