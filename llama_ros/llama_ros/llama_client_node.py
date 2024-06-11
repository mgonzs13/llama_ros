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
from typing import Callable, Tuple
from threading import Thread, RLock, Event

from rclpy.node import Node
from rclpy.client import Client
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from llama_msgs.srv import Tokenize
from llama_msgs.srv import GenerateEmbeddings
from llama_msgs.action import GenerateResponse


class LlamaClientNode(Node):

    _instance: "LlamaClientNode" = None
    _lock: RLock = RLock()

    _action_client: ActionClient = None
    _tokenize_srv_client: Client = None
    _embeddings_srv_client: Client = None

    _action_done_event: Event = Event()

    _action_result: GenerateResponse.Result
    _action_status: GoalStatus
    _goal_handle: ClientGoalHandle
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

        # executor
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = Thread(target=self._executor.spin)
        self._spin_thread.start()

    def tokenize(self, req: Tokenize.Request) -> Tokenize.Response:
        return self._tokenize_srv_client.call(req)

    def generate_embeddings(self, req: GenerateEmbeddings.Request) -> GenerateEmbeddings.Response:
        return self._embeddings_srv_client.call(req)

    def generate_response(self, goal: GenerateResponse.Goal, feedback_cb: Callable = None) -> Tuple[GenerateResponse.Result, GoalStatus]:

        self._action_client.wait_for_server()

        if feedback_cb is None:
            feedback_cb = self._feedback_callback

        self._action_done_event.clear()
        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback_cb)
        send_goal_future.add_done_callback(self._goal_response_callback)

        # Wait for action to be done
        self._action_done_event.wait()

        with self._goal_handle_lock:
            self._goal_handle = None

        return self._action_result, self._action_status

    def _goal_response_callback(self, future) -> None:

        with self._goal_handle_lock:
            self._goal_handle = future.result()
            get_result_future = self._goal_handle.get_result_async()
            get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future) -> None:
        self._action_result: GenerateResponse.Result = future.result().result
        self._action_status = future.result().status
        self._action_done_event.set()

    def _feedback_callback(self, feedback) -> None:
        pass

    def cancel_generate_text(self) -> None:
        with self._goal_handle_lock:
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal()
