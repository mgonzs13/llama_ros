# Copyright (C) 2024  Miguel Ángel González Santamarta

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

    _action_result: GenerateResponse.Result
    _action_status: GoalStatus
    _goal_handle: ClientGoalHandle
    _goal_handle_lock: RLock = RLock()

    _callback_group: ReentrantCallbackGroup = ReentrantCallbackGroup()
    _executor: MultiThreadedExecutor = None
    _spin_thread: Thread = None
    _action_done_event = Event()

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

    def generate_response(self, goal: GenerateResponse.Goal, feedback_cb: Callable = None) -> Tuple[GenerateResponse.Result | GoalStatus]:

        self._action_client.wait_for_server()

        if feedback_cb is None:
            feedback_cb = self._feedback_callback

        self._action_done_event.clear()
        send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=feedback_cb)

        with self._goal_handle_lock:
            self._goal_handle = send_goal_future.result()

        send_goal_future.add_done_callback(self._goal_response_callback)

        # Wait for action to be done
        self._action_done_event.wait()

        with self._goal_handle_lock:
            self._goal_handle = None

        return self._action_result, self._action_status

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        get_result_future = goal_handle.get_result_async()
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
