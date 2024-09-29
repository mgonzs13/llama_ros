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


import uuid
from typing import Callable, Tuple, List, Union, Generator
from threading import Thread, RLock, Condition

from rclpy.node import Node
from rclpy.client import Client
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from llama_msgs.srv import Tokenize
from llama_msgs.srv import GenerateEmbeddings
from llama_msgs.srv import RerankDocuments
from llama_msgs.srv import FormatChatMessages
from llama_msgs.action import GenerateResponse
from llama_msgs.msg import PartialResponse


class LlamaClientNode(Node):

    _instance: "LlamaClientNode" = None
    _lock: RLock = RLock()

    _action_client: ActionClient = None
    _tokenize_srv_client: Client = None
    _embeddings_srv_client: Client = None

    _action_done: bool = False
    _action_done_cond: Condition = Condition()

    _action_result: GenerateResponse.Result = None
    _action_status: GoalStatus = GoalStatus.STATUS_UNKNOWN
    _partial_results: List[PartialResponse] = []
    _goal_handle: ClientGoalHandle = None
    _goal_handle_lock: RLock = RLock()

    _callback_group: ReentrantCallbackGroup = ReentrantCallbackGroup()
    _executor: MultiThreadedExecutor = None
    _spin_thread: Thread = None

    @staticmethod
    def get_instance(namespace: str = "llama") -> "LlamaClientNode":

        with LlamaClientNode._lock:
            if LlamaClientNode._instance == None:
                LlamaClientNode._instance = LlamaClientNode(namespace)

            return LlamaClientNode._instance

    def __init__(self, namespace: str = "llama") -> None:

        if not LlamaClientNode._instance is None:
            raise Exception("This class is a Singleton")

        super().__init__(
            f"client_{str(uuid.uuid4()).replace('-', '_')}_node", namespace=namespace)

        self._action_client = ActionClient(
            self,
            GenerateResponse,
            "generate_response",
            callback_group=self._callback_group
        )

        self._tokenize_srv_client = self.create_client(
            Tokenize,
            "tokenize",
            callback_group=self._callback_group
        )

        self._embeddings_srv_client = self.create_client(
            GenerateEmbeddings,
            "generate_embeddings",
            callback_group=self._callback_group
        )

        self._rerank_srv_client = self.create_client(
            RerankDocuments,
            "rerank_documents",
            callback_group=self._callback_group
        )

        self._format_chat_srv_client = self.create_client(
            FormatChatMessages,
            "format_chat_prompt",
            callback_group=self._callback_group
        )

        # executor
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = Thread(target=self._executor.spin)
        self._spin_thread.start()

    def tokenize(self, req: Tokenize.Request) -> Tokenize.Response:
        self._tokenize_srv_client.wait_for_service()
        return self._tokenize_srv_client.call(req)

    def generate_embeddings(self, req: GenerateEmbeddings.Request) -> GenerateEmbeddings.Response:
        self._embeddings_srv_client.wait_for_service()
        return self._embeddings_srv_client.call(req)

    def rerank_documents(self, req: RerankDocuments.Request) -> RerankDocuments.Response:
        self._rerank_srv_client.wait_for_service()
        return self._rerank_srv_client.call(req)

    def format_chat_prompt(self, req: FormatChatMessages.Request) -> FormatChatMessages.Response:
        self._format_chat_srv_client.wait_for_service()
        return self._format_chat_srv_client.call(req)

    def generate_response(
        self,
        goal: GenerateResponse.Goal,
        feedback_cb: Callable = None,
        stream: bool = False
    ) -> Union[Tuple[GenerateResponse.Result, GoalStatus], Generator[PartialResponse, None, None]]:

        self._action_done = False
        self._action_result = None
        self._action_status = GoalStatus.STATUS_UNKNOWN
        self._partial_results = []
        self._action_client.wait_for_server()

        if feedback_cb is None and stream:
            feedback_cb = self._feedback_callback

        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback_cb)
        send_goal_future.add_done_callback(self._goal_response_callback)

        # Wait for action to be done
        def generator():
            with self._action_done_cond:
                while not self._action_done:

                    while self._partial_results:
                        yield self._partial_results.pop(0)

                    self._action_done_cond.wait()

            if self._partial_results:
                yield from self._partial_results

        if stream:
            return generator()

        else:
            with self._action_done_cond:
                while not self._action_done:
                    self._action_done_cond.wait()
            return self._action_result, self._action_status

    def _goal_response_callback(self, future) -> None:

        with self._goal_handle_lock:
            self._goal_handle = future.result()
            get_result_future = self._goal_handle.get_result_async()
            get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future) -> None:

        self._action_result: GenerateResponse.Result = future.result().result
        self._action_status = future.result().status

        with self._action_done_cond:
            self._action_done = True
            self._action_done_cond.notify()

        with self._goal_handle_lock:
            self._goal_handle = None

    def _feedback_callback(self, feedback) -> None:
        self._partial_results.append(feedback.feedback.partial_response)

        with self._action_done_cond:
            self._action_done_cond.notify()

    def cancel_generate_text(self) -> None:
        with self._goal_handle_lock:
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal()
