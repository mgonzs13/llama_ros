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


from typing import Dict, List
from pydantic import BaseModel, Extra, root_validator
from langchain_core.embeddings import Embeddings

from simple_node import Node
from rclpy.client import Client
from llama_msgs.srv import GenerateEmbeddings


class LlamaROSEmbeddings(BaseModel, Embeddings):

    node: Node

    generate_embeddings_srv_name: str = "/llama/generate_embeddings"
    generate_embeddings_srv_client: Client = None

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:

        node: Node = values["node"]

        generate_embeddings_srv_name = values["generate_embeddings_srv_name"]
        values["generate_embeddings_srv_client"] = node.create_client(
            GenerateEmbeddings, generate_embeddings_srv_name)

        return values

    def __call_generate_embedding_srv(self, text: str) -> List[int]:

        req = GenerateEmbeddings.Request()
        req.prompt = text

        self.generate_embeddings_srv_client.wait_for_service()
        res = self.generate_embeddings_srv_client.call(req)

        return res.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = [self.__call_generate_embedding_srv(
            text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.__call_generate_embedding_srv(text)
        return list(map(float, embedding))
