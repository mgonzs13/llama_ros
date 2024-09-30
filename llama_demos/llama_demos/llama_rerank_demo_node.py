#!/usr/bin/env python3

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


import operator
import rclpy
from rclpy.node import Node
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.srv import RerankDocuments


class LlamaRerankDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("llama_rerank_demo_node")

        self._llama_client = LlamaClientNode.get_instance()

    def send_rerank(self) -> None:

        rerank_req = RerankDocuments.Request()
        rerank_req.query = "Machine learning is"
        rerank_req.documents = [
            "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
            "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
            "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
            "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
        ]

        ranks = self._llama_client.rerank_documents(rerank_req).ranks
        scores = [r.score for r in ranks]

        docs_with_scores = list(zip(ranks, scores))
        result = sorted(docs_with_scores,
                        key=operator.itemgetter(1), reverse=True)

        for i in range(len(result)):
            self.get_logger().info(
                f"{i} ({result[i][0].score}): {result[i][0].document}")


def main():
    rclpy.init()
    node = LlamaRerankDemoNode()
    node.send_rerank()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
