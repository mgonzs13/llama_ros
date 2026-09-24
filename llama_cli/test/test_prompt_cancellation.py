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


"""Portable tests of the CLI and its shared response request implementation."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import sys
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Goal:
    def __init__(self):
        self.sampling_config = SimpleNamespace(temp=0.0)
        self.images = []


class PromptTests(unittest.TestCase):
    def setUp(self):
        names = (
            "rclpy",
            "rclpy.node",
            "rclpy.client",
            "rclpy.action",
            "rclpy.action.client",
            "rclpy.callback_groups",
            "rclpy.executors",
            "rclpy.signals",
            "action_msgs",
            "action_msgs.msg",
            "llama_msgs",
            "llama_msgs.srv",
            "llama_msgs.action",
            "llama_msgs.msg",
            "launch",
            "launch_ros",
            "launch_ros.actions",
            "cv2",
            "cv_bridge",
            "numpy",
            "yaml",
        )
        self.modules = {name: MagicMock() for name in names}
        self.modules["rclpy.node"].Node = object
        self.modules["action_msgs.msg"].GoalStatus = SimpleNamespace(
            STATUS_UNKNOWN=0, STATUS_SUCCEEDED=4
        )
        self.modules["llama_msgs.action"].GenerateResponse = SimpleNamespace(
            Goal=Goal, Result=object
        )
        self.modules["llama_msgs.action"].GenerateChatCompletions = SimpleNamespace(
            Goal=Goal, Result=object, Feedback=object
        )
        self.modules["llama_msgs.msg"].PartialResponse = object
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.client_module = load_module(
            "tested_llama_client", ROOT / "llama_ros/llama_ros/llama_client_node.py"
        )
        self.modules["llama_ros.llama_client_node"] = self.client_module
        self.client_patch = patch.dict(
            sys.modules, {"llama_ros.llama_client_node": self.client_module}
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.api = load_module("tested_api", ROOT / "llama_cli/llama_cli/api/__init__.py")
        self.request = self.client_module.ResponseRequest()
        self.acceptance = Future()
        self.acceptance.add_done_callback(self.request._accepted)
        self.result_future = Future()
        self.handle = SimpleNamespace(
            accepted=True,
            get_result_async=lambda: self.result_future,
            cancel_goal_async=MagicMock(),
        )
        self.client = MagicMock()
        self.client.wait_for_response_server.return_value = True
        self.client.generate_response_async.side_effect = self.submit
        self.api.LlamaClientNode = MagicMock(return_value=self.client)
        self.on_submit = lambda: self.complete()

    def complete(self, status=4, text=""):
        self.acceptance.set_result(self.handle)
        self.result_future.set_result(
            SimpleNamespace(
                status=status, result=SimpleNamespace(response=SimpleNamespace(text=text))
            )
        )

    def submit(self, goal, feedback_cb):
        self.goal = goal
        self.feedback = feedback_cb
        self.on_submit()
        return self.request

    def run_prompt(self, **kwargs):
        output, errors = io.StringIO(), io.StringIO()
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = self.api.prompt_llm("hello", **kwargs)
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)
        return result, output.getvalue(), errors.getvalue()

    def pipe(self):
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        return reader, writer

    def test_precompute_success_is_silent(self):
        result, output, _ = self.run_prompt(precompute=True, action_name="/custom")
        self.assertEqual(result, 0)
        self.assertEqual(output, "")
        self.assertTrue(self.goal.precompute)
        self.api.LlamaClientNode.assert_called_once_with(action_name="/custom")
        self.client.close.assert_called_once()

    def test_normal_response_appends_missing_feedback_suffix(self):
        def submit():
            self.feedback(
                SimpleNamespace(
                    feedback=SimpleNamespace(partial_response=SimpleNamespace(text="hel"))
                )
            )
            self.complete(text="hello")

        self.on_submit = submit
        result, output, _ = self.run_prompt()
        self.assertEqual((result, output), (0, "hello\n"))
        self.assertFalse(self.goal.precompute)

    def test_rejected_goal_fails(self):
        self.handle.accepted = False
        self.on_submit = lambda: self.acceptance.set_result(self.handle)
        result, _, error = self.run_prompt()
        self.assertEqual(result, 1)
        self.assertIn("rejected", error)

    def test_non_success_terminal_statuses_fail(self):
        for status in (5, 6):
            request = self.client_module.ResponseRequest()
            request.status = status
            request.done.set()
            self.client.generate_response_async.side_effect = lambda *a, **k: request
            self.assertEqual(self.run_prompt()[0], 1)

    def test_eof_before_submission(self):
        reader, writer = self.pipe()
        os.close(writer)
        self.assertEqual(self.run_prompt(cancel_fd=reader)[0], 1)
        self.client.generate_response_async.assert_not_called()

    def test_eof_before_acceptance_cancels_late_handle(self):
        reader, writer = self.pipe()
        self.on_submit = lambda: os.close(writer)

        def wait(timeout):
            self.assertTrue(self.request._cancel_requested)
            self.acceptance.set_result(self.handle)
            self.handle.cancel_goal_async.assert_called_once()
            self.result_future.set_result(SimpleNamespace(status=5, result=None))

        self.request.done.wait = wait
        self.assertEqual(self.run_prompt(cancel_fd=reader)[0], 1)

    def test_sigint_and_sigterm_cancel_accepted_goal(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                self.request = self.client_module.ResponseRequest()
                accepted, result = Future(), Future()
                handle = SimpleNamespace(
                    accepted=True,
                    get_result_async=lambda: result,
                    cancel_goal_async=MagicMock(
                        side_effect=lambda: result.set_result(
                            SimpleNamespace(status=5, result=None)
                        )
                    ),
                )

                def submit():
                    accepted.add_done_callback(self.request._accepted)
                    accepted.set_result(handle)
                    signal.getsignal(signum)(signum, None)

                self.on_submit = submit
                self.assertEqual(self.run_prompt()[0], 1)
                handle.cancel_goal_async.assert_called_once()

    def test_cancellation_has_one_deadline_during_acceptance(self):
        reader, writer = self.pipe()
        self.on_submit = lambda: os.close(writer)
        self.request.done.wait = lambda timeout: self.acceptance.set_result(self.handle)
        with patch.object(self.api.time, "monotonic", side_effect=[10, 10, 16]):
            result, _, error = self.run_prompt(cancel_fd=reader)
        self.assertEqual(result, 1)
        self.assertIn("Timed out", error)
        self.handle.cancel_goal_async.assert_called_once()

    def test_success_racing_with_eof_still_fails(self):
        reader, writer = self.pipe()

        def submit():
            self.complete()
            os.close(writer)

        self.on_submit = submit
        self.assertEqual(self.run_prompt(cancel_fd=reader)[0], 1)

    def test_partial_signal_installation_restores_first_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        real_signal = signal.signal

        def install(signum, handler):
            if signum == signal.SIGTERM:
                raise ValueError("signal setup failed")
            return real_signal(signum, handler)

        with patch.object(self.api.signal, "signal", side_effect=install):
            self.assertEqual(self.run_prompt()[0], 1)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)
        self.modules["rclpy"].init.assert_not_called()

    def test_request_cancellation_does_not_touch_other_request(self):
        other = self.client_module.ResponseRequest()
        other_future, other_result = Future(), Future()
        other_handle = SimpleNamespace(
            accepted=True,
            get_result_async=lambda: other_result,
            cancel_goal_async=MagicMock(),
        )
        other_future.add_done_callback(other._accepted)
        other_future.set_result(other_handle)
        self.request.cancel()
        self.acceptance.set_result(self.handle)
        self.request.cancel()
        self.handle.cancel_goal_async.assert_called_once()
        other_handle.cancel_goal_async.assert_not_called()

    def test_feedback_error_cancels_through_shared_client(self):
        client = object.__new__(self.client_module.LlamaClientNode)
        client._action_client = MagicMock()
        accepted = Future()
        client._action_client.send_goal_async.return_value = accepted
        callback = MagicMock(side_effect=OSError("output closed"))
        request = client.generate_response_async(Goal(), feedback_cb=callback)
        feedback = client._action_client.send_goal_async.call_args.kwargs[
            "feedback_callback"
        ]
        feedback(object())
        self.assertIn("output closed", str(request.error))
        accepted.set_result(self.handle)
        self.handle.cancel_goal_async.assert_called_once()

    def test_feedback_error_wait_is_bounded(self):
        self.on_submit = lambda: setattr(self.request, "error", OSError("output closed"))
        self.request.done.wait = lambda timeout: None
        with patch.object(self.api.time, "monotonic", side_effect=[10, 10, 16]):
            result, _, error = self.run_prompt()
        self.assertEqual(result, 1)
        self.assertIn("Timed out", error)

    def test_cleanup_exception_still_restores_handlers(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        self.client.close.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            self.run_prompt(precompute=True)
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)
        self.modules["rclpy"].shutdown.assert_called_once()

    def test_multimodal_goal_keeps_image(self):
        image = object()
        self.api.CvBridge.return_value.cv2_to_imgmsg.return_value = image
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"image"
        with patch.object(self.api.urllib.request, "urlopen", return_value=response):
            self.assertEqual(self.run_prompt(image_url="https://example.org/image")[0], 0)
        self.assertEqual(self.goal.images, [image])

    def test_acceptance_exception_finishes_request(self):
        self.acceptance.set_exception(RuntimeError("transport error"))
        self.assertTrue(self.request.done.is_set())
        self.assertIn("transport error", str(self.request.error))


if __name__ == "__main__":
    unittest.main()
