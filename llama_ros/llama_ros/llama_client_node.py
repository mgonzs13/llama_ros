# MIT License
#
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


import uuid
from typing import Callable, Tuple, List, Union, Generator
from threading import Thread, RLock, Condition, Event

from rclpy.node import Node
from rclpy.client import Client
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from llama_msgs.srv import GetMetadata
from llama_msgs.srv import Tokenize
from llama_msgs.srv import Detokenize
from llama_msgs.srv import GenerateEmbeddings
from llama_msgs.srv import RerankDocuments
from llama_msgs.action import GenerateResponse
from llama_msgs.action import GenerateChatCompletions
from llama_msgs.msg import PartialResponse


class ResponseRequest:
    """Track one response, including cancellation before acceptance."""

    def __init__(self):
        self.done = Event()
        self.result = None
        self.status = GoalStatus.STATUS_UNKNOWN
        self.error = None
        self._lock = RLock()
        self._handle = None
        self._cancel_requested = False
        self._cancel_sent = False

    def cancel(self):
        """Remember cancellation until the server supplies this request's handle."""
        with self._lock:
            self._cancel_requested = True
            if (
                self._handle is not None
                and not self._cancel_sent
                and not self.done.is_set()
            ):
                self._cancel_sent = True
                self._handle.cancel_goal_async()

    def _accepted(self, future):
        with self._lock:
            try:
                self._handle = future.result()
                if not self._handle.accepted:
                    self.error = RuntimeError("Goal was rejected by the action server")
                    self.done.set()
                    return
                self._handle.get_result_async().add_done_callback(self._completed)
                if self._cancel_requested:
                    self.cancel()
            except Exception as exc:
                self.error = exc
                self.done.set()

    def _completed(self, future):
        with self._lock:
            try:
                response = future.result()
                self.result = response.result
                self.status = response.status
            except Exception as exc:
                self.error = exc
            finally:
                self.done.set()


class LlamaClientNode(Node):

    _instance: "LlamaClientNode" = None
    _lock: RLock = RLock()

    _action_client: ActionClient = None
    _action_chat_client: ActionClient = None
    _tokenize_srv_client: Client = None
    _embeddings_srv_client: Client = None

    _action_done: bool = False
    _action_done_cond: Condition = Condition()

    _action_result = None
    _action_status: GoalStatus = GoalStatus.STATUS_UNKNOWN
    _partial_results: List[PartialResponse | GenerateChatCompletions.Feedback] = []
    _goal_handle: ClientGoalHandle = None
    _goal_handle_lock: RLock = RLock()

    _callback_group: ReentrantCallbackGroup = ReentrantCallbackGroup()
    _executor: MultiThreadedExecutor = None
    _spin_thread: Thread = None

    @staticmethod
    def get_instance() -> "LlamaClientNode":

        with LlamaClientNode._lock:
            if LlamaClientNode._instance == None:
                LlamaClientNode._instance = LlamaClientNode()

            return LlamaClientNode._instance

    def __init__(
        self, namespace: str = "llama", action_name: str = "generate_response"
    ) -> None:

        if not LlamaClientNode._instance is None:
            raise Exception("This class is a Singleton")

        super().__init__(
            f"client_{str(uuid.uuid4()).replace('-', '_')}_node", namespace=namespace
        )

        self._get_metadata_srv_client = self.create_client(
            GetMetadata, "get_metadata", callback_group=self._callback_group
        )

        self._tokenize_srv_client = self.create_client(
            Tokenize, "tokenize", callback_group=self._callback_group
        )

        self._detokenize_srv_client = self.create_client(
            Detokenize, "detokenize", callback_group=self._callback_group
        )

        self._embeddings_srv_client = self.create_client(
            GenerateEmbeddings,
            "generate_embeddings",
            callback_group=self._callback_group,
        )

        self._rerank_srv_client = self.create_client(
            RerankDocuments, "rerank_documents", callback_group=self._callback_group
        )

        self._action_client = ActionClient(
            self,
            GenerateResponse,
            action_name,
            callback_group=self._callback_group,
        )

        self._action_chat_client = ActionClient(
            self,
            GenerateChatCompletions,
            "generate_chat_completions",
            callback_group=self._callback_group,
        )

        # executor
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = Thread(target=self._executor.spin)
        self._spin_thread.start()

    def wait_for_response_server(self, timeout_sec: float = 0.1) -> bool:
        """Wait briefly for the response action without blocking cancellation."""
        return self._action_client.wait_for_server(timeout_sec=timeout_sec)

    def generate_response_async(
        self, goal: GenerateResponse.Goal, feedback_cb: Callable = None
    ) -> ResponseRequest:
        """Submit independently tracked work after checking server readiness."""
        request = ResponseRequest()

        def feedback(message):
            try:
                feedback_cb(message)
            except Exception as exc:
                request.error = exc
                request.cancel()

        future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback if feedback_cb is not None else None
        )
        future.add_done_callback(request._accepted)
        return request

    def close(self) -> None:
        """Stop the executor and release this client's ROS resources."""
        self._executor.shutdown(timeout_sec=1.0, wait_for_threads=False)
        self._spin_thread.join(timeout=1.0)
        self.destroy_node()
        with self._lock:
            if LlamaClientNode._instance is self:
                LlamaClientNode._instance = None

    def get_metadata(self, req: GetMetadata.Request) -> GetMetadata:
        self._get_metadata_srv_client.wait_for_service()
        return self._get_metadata_srv_client.call(req)

    def tokenize(self, req: Tokenize.Request) -> Tokenize.Response:
        self._tokenize_srv_client.wait_for_service()
        return self._tokenize_srv_client.call(req)

    def detokenize(self, req: Detokenize.Request) -> Detokenize.Response:
        self._detokenize_srv_client.wait_for_service()
        return self._detokenize_srv_client.call(req)

    def generate_embeddings(
        self, req: GenerateEmbeddings.Request
    ) -> GenerateEmbeddings.Response:
        self._embeddings_srv_client.wait_for_service()
        return self._embeddings_srv_client.call(req)

    def rerank_documents(self, req: RerankDocuments.Request) -> RerankDocuments.Response:
        self._rerank_srv_client.wait_for_service()
        return self._rerank_srv_client.call(req)

    def generate_chat_completions(
        self,
        goal: GenerateChatCompletions.Goal,
        feedback_cb: Callable = None,
        stream: bool = False,
        stream_reasoning: bool = False,
    ) -> Union[
        Tuple[GenerateChatCompletions.Result, GoalStatus],
        Generator[GenerateChatCompletions.Feedback, None, None],
    ]:
        self._action_done = False
        self._action_result = None
        self._action_status = GoalStatus.STATUS_UNKNOWN
        self._partial_results = []
        self._action_chat_client.wait_for_server()

        if feedback_cb is None and stream:
            feedback_cb = self._feedback_callback_chat

        send_goal_future = self._action_chat_client.send_goal_async(
            goal, feedback_callback=feedback_cb
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

        # Wait for action to be done
        def generator():
            while not self._action_done:
                # Collect and yield any available results
                with self._action_done_cond:
                    to_yield = self._partial_results[:]
                    self._partial_results.clear()

                for item in to_yield:
                    if stream_reasoning and item.choices[0].delta.reasoning_content:
                        item.choices[0].delta.content = item.choices[
                            0
                        ].delta.reasoning_content
                    yield item

                # Wait for more results or completion
                if not self._action_done:
                    with self._action_done_cond:
                        if not self._partial_results and not self._action_done:
                            self._action_done_cond.wait()

            # Yield any final results
            with self._action_done_cond:
                for item in self._partial_results:
                    if stream_reasoning and item.choices[0].delta.reasoning_content:
                        item.choices[0].delta.content = item.choices[
                            0
                        ].delta.reasoning_content
                    yield item

        if stream:
            return generator()

        else:
            with self._action_done_cond:
                while not self._action_done:
                    self._action_done_cond.wait()
            return self._action_result, self._action_status

    def generate_response(
        self,
        goal: GenerateResponse.Goal,
        feedback_cb: Callable = None,
        stream: bool = False,
    ) -> Union[
        Tuple[GenerateResponse.Result, GoalStatus],
        Generator[PartialResponse, None, None],
    ]:

        self._action_done = False
        self._action_result = None
        self._action_status = GoalStatus.STATUS_UNKNOWN
        self._partial_results = []
        self._action_client.wait_for_server()

        if feedback_cb is None and stream:
            feedback_cb = self._feedback_callback

        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback_cb
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

        # Wait for action to be done
        def generator():
            while not self._action_done:
                # Collect and yield any available results
                with self._action_done_cond:
                    to_yield = self._partial_results[:]
                    self._partial_results.clear()

                for item in to_yield:
                    yield item

                # Wait for more results or completion
                if not self._action_done:
                    with self._action_done_cond:
                        if not self._partial_results and not self._action_done:
                            self._action_done_cond.wait()

            # Yield any final results
            with self._action_done_cond:
                for item in self._partial_results:
                    yield item

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
            if not self._goal_handle.accepted:
                self.get_logger().error("Goal was REJECTED by the action server.")
                self._goal_handle = None
                with self._action_done_cond:
                    self._action_done = True
                    self._action_done_cond.notify_all()
                return
            self.get_logger().debug("Goal accepted by action server.")
            get_result_future = self._goal_handle.get_result_async()
            get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future) -> None:

        self._action_result: GenerateResponse.Result = future.result().result
        self._action_status = future.result().status

        _STATUS_NAMES = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        status_name = _STATUS_NAMES.get(
            self._action_status, f"UNKNOWN({self._action_status})"
        )
        if self._action_status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"Action finished with non-success status: {status_name} "
                f"(code={self._action_status}). "
                "Check llama_node logs for the RCLCPP_ERROR above this."
            )
        else:
            self.get_logger().debug(f"Action finished with status: {status_name}")

        with self._action_done_cond:
            self._action_done = True
            self._action_done_cond.notify()

        with self._goal_handle_lock:
            self._goal_handle = None

    def _feedback_callback_chat(self, feedback) -> None:
        self._partial_results.append(feedback.feedback)

        with self._action_done_cond:
            self._action_done_cond.notify()

    def _feedback_callback(self, feedback) -> None:
        self._partial_results.append(feedback.feedback.partial_response)

        with self._action_done_cond:
            self._action_done_cond.notify()

    def cancel_generate_text(self) -> None:
        with self._goal_handle_lock:
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal()
