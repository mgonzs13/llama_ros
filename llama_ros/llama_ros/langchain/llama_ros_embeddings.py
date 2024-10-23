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


from typing import Dict, List
from pydantic import BaseModel, model_validator
from langchain_core.embeddings import Embeddings

from llama_msgs.srv import GenerateEmbeddings
from llama_ros.llama_client_node import LlamaClientNode


class LlamaROSEmbeddings(BaseModel, Embeddings):

    llama_client: LlamaClientNode = None
    normalization: int = 2

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        values["llama_client"] = LlamaClientNode.get_instance()
        return values

    def __call_generate_embedding_srv(self, text: str) -> List[int]:
        req = GenerateEmbeddings.Request()
        req.prompt = text
        req.normalization = self.normalization
        return self.llama_client.generate_embeddings(req).embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = [self.__call_generate_embedding_srv(
            text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.__call_generate_embedding_srv(text)
        return list(map(float, embedding))
