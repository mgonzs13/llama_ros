# MIT License

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


from typing import Any, Dict, List, Optional, Iterator

from action_msgs.msg import GoalStatus
from llama_msgs.srv import Tokenize
from llama_ros.langchain import LlamaROSCommon

from langchain_core.outputs import GenerationChunk
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class LlamaROS(LLM, LlamaROSCommon):

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "llamaros"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        goal = self._create_action_goal(prompt, stop, **kwargs)

        result, status = self.llama_client.generate_response(goal)

        if status != GoalStatus.STATUS_SUCCEEDED:
            return ""
        return result.response.text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        goal = self._create_action_goal(prompt, stop, **kwargs)

        for pt in self.llama_client.generate_response(goal, stream=True):

            if run_manager:
                run_manager.on_llm_new_token(
                    pt.text,
                    verbose=self.verbose,
                )

            yield GenerationChunk(text=pt.text)

    def get_num_tokens(self, text: str) -> int:
        req = Tokenize.Request()
        req.text = text
        tokens = self.llama_client.tokenize(req).tokens
        return len(tokens)
