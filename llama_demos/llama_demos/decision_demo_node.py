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
from llama_msgs.msg import DecisionQuestion
from llama_msgs.srv import EvaluateDecisions

SCENARIOS = [
    {
        "name": "low battery, full dustbin",
        "hint": "The battery is low and the dustbin is full.",
        "state": {
            "battery": "low (5%)",
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


def evaluate(llama_client, state, questions):
    req = EvaluateDecisions.Request()
    req.state = json.dumps(state)
    req.questions = questions
    return llama_client.evaluate_decisions(req)


def print_answer(name, answer):
    if not answer.success:
        print(f"  {name}: failed: {answer.error}")
        return

    if answer.type == DecisionQuestion.CHOICE:
        print(f"  {name}: {answer.choice} (confidence {answer.confidence:.3f})")
    elif answer.type == DecisionQuestion.SCORE:
        print(f"  {name}: {answer.score:.2f} (confidence {answer.confidence:.3f})")
    else:
        print(f"  {name}: P(true) = {answer.noul:.3f}")

    for key, probability in zip(answer.keys, answer.probabilities):
        print(f"    {key}: {probability:.3f}")


def run_scenario(llama_client, scenario):
    state = scenario["state"]
    hint = scenario["hint"]

    print(f"=== {scenario['name']}")
    print(f"  state: {json.dumps(state)}")

    choice = DecisionQuestion()
    choice.type = DecisionQuestion.CHOICE
    choice.instructions = f"{hint} Choose the best next action for the robot."
    choice.keys = CHOICE_KEYS
    choice.descriptions = CHOICE_DESCRIPTIONS

    score = DecisionQuestion()
    score.type = DecisionQuestion.SCORE
    score.instructions = f"{hint} How urgent is it to recharge the battery now?"
    score.descriptions = SCORE_DESCRIPTIONS

    noul = DecisionQuestion()
    noul.type = DecisionQuestion.NOUL
    noul.instructions = (
        f"{hint} The battery is sufficient to keep cleaning for at least "
        "30 more minutes."
    )

    response = evaluate(llama_client, state, [choice, score, noul])

    if len(response.answers) != 3:
        print(f"  expected 3 answers, got {len(response.answers)}")
        return

    print_answer("choice", response.answers[0])
    print_answer("score", response.answers[1])
    print_answer("noul", response.answers[2])


def main():
    rclpy.init()

    llama_client = LlamaClientNode.get_instance()

    for scenario in SCENARIOS:
        run_scenario(llama_client, scenario)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
