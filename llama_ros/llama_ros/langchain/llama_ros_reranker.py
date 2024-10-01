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


from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.pydantic_v1 import Extra
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from llama_msgs.srv import RerankDocuments
from llama_ros.llama_client_node import LlamaClientNode


class LlamaROSReranker(BaseDocumentCompressor):

    top_n: int = 3

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:

        req = RerankDocuments.Request()
        req.query = query

        for doc in documents:
            req.documents.append(doc.page_content)

        scores = LlamaClientNode.get_instance().rerank_documents(req).scores
        scored_docs = list(zip(documents, scores))
        result = sorted(scored_docs, key=operator.itemgetter(1), reverse=True)
        return [doc for doc, _ in result[: self.top_n]]
