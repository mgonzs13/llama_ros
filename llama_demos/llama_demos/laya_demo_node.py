#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Miguel Ángel González Santamarta
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


import json

import rclpy
from llama_ros.llama_client_node import LlamaClientNode
from llama_msgs.srv import EvaluateDecision

# Laya is trained on typed/structured states with qualitative labels, so the
# state is given as JSON and the key facts are also spelled out in the
# instructions.
SCENARIOS = [
    {
        "name": "low battery, full dustbin",
        "hint": "The battery is low and the dustbin is full.",
        "state": {
            "battery": "low (12%)",
            "dustbin": "full",
            "dock_distance_m": 4,
            "task": "cleaning the living room",
        },
    },
    {
        "name": "high battery, empty dustbin",
        "hint": "The battery is high and the dustbin is empty.",
        "state": {
            "battery": "high (90%)",
            "dustbin": "empty",
            "dock_distance_m": 4,
            "task": "cleaning the living room",
        },
    },
]

CHOICE_KEYS = ["continue", "return_to_dock", "stop"]
CHOICE_DESCRIPTIONS = [
    "Keep cleaning the current area.",
    "Return to the docking station to recharge and empty the dustbin.",
    "Stop and wait for a human.",
]
SCORE_DESCRIPTIONS = ["not urgent", "slightly urgent", "urgent", "critical"]


def evaluate(
    llama_client, state, question_type, instructions, keys=None, descriptions=None
):
    req = EvaluateDecision.Request()
    req.type = question_type
    req.instructions = instructions
    req.state = json.dumps(state)
    req.keys = keys if keys is not None else []
    req.descriptions = descriptions if descriptions is not None else []

    return llama_client.evaluate_decision(req)


def print_answer(name, res):
    if not res.success:
        print(f"  {name}: failed: {res.error}")
        return

    if res.type == EvaluateDecision.Request.CHOICE:
        print(f"  {name}: {res.choice} (confidence {res.confidence:.3f})")
    elif res.type == EvaluateDecision.Request.SCORE:
        print(f"  {name}: {res.score:.2f} (confidence {res.confidence:.3f})")
    else:
        print(f"  {name}: P(true) = {res.noul:.3f}")

    for key, probability in zip(res.keys, res.probabilities):
        print(f"    {key}: {probability:.3f}")


def run_scenario(llama_client, scenario):
    state = scenario["state"]
    hint = scenario["hint"]

    print(f"=== {scenario['name']}")
    print(f"  state: {json.dumps(state)}")

    choice = evaluate(
        llama_client,
        state,
        EvaluateDecision.Request.CHOICE,
        f"{hint} Choose the best next action for the robot.",
        keys=CHOICE_KEYS,
        descriptions=CHOICE_DESCRIPTIONS,
    )
    print_answer("choice", choice)

    score = evaluate(
        llama_client,
        state,
        EvaluateDecision.Request.SCORE,
        f"{hint} How urgent is it to recharge the battery now?",
        descriptions=SCORE_DESCRIPTIONS,
    )
    print_answer("score", score)

    noul = evaluate(
        llama_client,
        state,
        EvaluateDecision.Request.NOUL,
        f"{hint} The battery is sufficient to keep cleaning for at least 30 more minutes.",
    )
    print_answer("noul", noul)


def main():
    rclpy.init()

    llama_client = LlamaClientNode.get_instance()

    for scenario in SCENARIOS:
        run_scenario(llama_client, scenario)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
