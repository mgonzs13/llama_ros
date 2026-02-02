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
import requests
import tempfile
import numpy as np
from abc import ABC
import urllib.request
from cv_bridge import CvBridge
from pydantic import model_validator
from typing import List, Optional, Dict, Union

from langchain_core.language_models import BaseLanguageModel

from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.action import GenerateResponse
from llama_msgs.srv import GetMetadata
from llama_msgs.msg import LogitBias
from llama_msgs.msg import Metadata
from llama_msgs.msg import SamplingConfig
from sensor_msgs.msg import Image


class LlamaROSCommon(BaseLanguageModel, ABC):

    llama_client: LlamaClientNode = None
    cv_bridge: CvBridge = CvBridge()
    model_metadata: Metadata = None
    stream_reasoning: bool = False

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
    top_n_sigma: float = -1.0
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

    samplers_sequence: str = "edskypmxt"

    grammar: str = ""
    grammar_schema: str = ""
    grammar_lazy: bool = False
    grammar_triggers: List[List[Union[int, str]]] = []
    preserved_tokens: List[int] = []

    enable_thinking: bool = False

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:

        values["llama_client"] = LlamaClientNode.get_instance()
        values["model_metadata"] = (
            values["llama_client"].get_metadata(GetMetadata.Request()).metadata
        )
        return values

    def cancel(self) -> None:
        self.llama_client.cancel_generate_text()

    def _get_image(self, image_url: str, image: np.ndarray) -> Image:
        if image_url and image is None:
            req = urllib.request.Request(
                image_url, 
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"}
            )
            response = urllib.request.urlopen(req)
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)

        return self.cv_bridge.cv2_to_imgmsg(image)

    def download_audio_to_tempfile(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name

    def read_mp3_as_uint8_array(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8)

    def _create_action_goal(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        image_url: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> GenerateResponse.Result:

        goal = GenerateResponse.Goal()
        goal.prompt = prompt
        goal.reset = True

        # load image
        if image_url or image is not None:
            goal.images.append(self._get_image(image_url, image))

        # add stop
        if stop:
            goal.stop = stop

        # sampling params
        goal.sampling_config = self._set_sampling_config()

        return goal

    def _set_sampling_config(self):
        sampling_config = SamplingConfig()
        sampling_config.n_prev = self.n_prev
        sampling_config.n_probs = self.n_probs
        sampling_config.min_keep = self.min_keep

        sampling_config.ignore_eos = self.ignore_eos
        for key in self.logit_bias:
            lb = LogitBias()
            lb.token = key
            lb.bias = self.logit_bias[key]
            sampling_config.logit_bias.data.append(lb)

        sampling_config.temp = self.temp
        sampling_config.dynatemp_range = self.dynatemp_range
        sampling_config.dynatemp_exponent = self.dynatemp_exponent

        sampling_config.top_k = self.top_k
        sampling_config.top_p = self.top_p
        sampling_config.min_p = self.min_p
        sampling_config.xtc_probability = self.xtc_probability
        sampling_config.xtc_threshold = self.xtc_threshold
        sampling_config.typical_p = self.typical_p

        sampling_config.penalty_last_n = self.penalty_last_n
        sampling_config.penalty_repeat = self.penalty_repeat
        sampling_config.penalty_freq = self.penalty_freq
        sampling_config.penalty_present = self.penalty_present

        sampling_config.dry_multiplier = self.dry_multiplier
        sampling_config.dry_base = self.dry_base
        sampling_config.dry_allowed_length = self.dry_allowed_length
        sampling_config.dry_penalty_last_n = self.dry_penalty_last_n
        sampling_config.dry_sequence_breakers = self.dry_sequence_breakers

        sampling_config.mirostat = self.mirostat
        sampling_config.mirostat_eta = self.mirostat_eta
        sampling_config.mirostat_tau = self.mirostat_tau

        sampling_config.samplers_sequence = self.samplers_sequence

        sampling_config.grammar = self.grammar
        sampling_config.grammar_schema = self.grammar_schema
        sampling_config.grammar_lazy = self.grammar_lazy
        sampling_config.grammar_triggers = self.grammar_triggers
        sampling_config.preserved_tokens = self.preserved_tokens

        return sampling_config
