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

from abc import ABC
from typing import List, Optional, Dict
from pydantic import model_validator
from cv_bridge import CvBridge
import numpy as np
import urllib.request
import cv2

from langchain_core.language_models import BaseLanguageModel

from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse
from llama_msgs.msg import LogitBias


class LlamaROSCommon(BaseLanguageModel, ABC):

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
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    typical_p: float = 1.00

    penalty_last_n: int = 64
    penalty_repeat: float = 1.00
    penalty_freq: float = 0.00
    penalty_present: float = 0.00

    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1
    dry_sequence_breakers: List[str] = ["\\n", ":", '\\"', "*"]

    mirostat: int = 0
    mirostat_eta: float = 0.10
    mirostat_tau: float = 5.0

    penalize_nl: bool = False

    samplers_sequence: str = "dkypmxt"

    grammar: str = ""
    grammar_schema: str = ""

    penalty_prompt_tokens: List[int] = []
    use_penalty_prompt_tokens: bool = False

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        values["llama_client"] = LlamaClientNode.get_instance()
        return values

    def cancel(self) -> None:
        self.llama_client.cancel_generate_text()

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
        if image_url or image is not None:

            if image_url and image is None:
                req = urllib.request.Request(
                    image_url, headers={"User-Agent": "Mozilla/5.0"}
                )
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
        goal.sampling_config.xtc_probability = self.xtc_probability
        goal.sampling_config.xtc_threshold = self.xtc_threshold
        goal.sampling_config.typical_p = self.typical_p

        goal.sampling_config.penalty_last_n = self.penalty_last_n
        goal.sampling_config.penalty_repeat = self.penalty_repeat
        goal.sampling_config.penalty_freq = self.penalty_freq
        goal.sampling_config.penalty_present = self.penalty_present

        goal.sampling_config.dry_multiplier = self.dry_multiplier
        goal.sampling_config.dry_base = self.dry_base
        goal.sampling_config.dry_allowed_length = self.dry_allowed_length
        goal.sampling_config.dry_penalty_last_n = self.dry_penalty_last_n
        goal.sampling_config.dry_sequence_breakers = self.dry_sequence_breakers

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
