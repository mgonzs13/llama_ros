#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024 Alejandro González Cantón
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


import time
import rclpy
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llama_ros.langchain import ChatLlamaROS


def main():

    tokens = 0
    initial_time = -1
    eval_time = -1

    rclpy.init()
    chat = ChatLlamaROS(
        temp=0.2,
        penalty_repeat=1.15,
        enable_thinking=True,
        stream_reasoning=True,
    )

    domain = """(define (domain exercise0)
  (:requirements :strips :typing :negative-preconditions :disjunctive-preconditions :equality)

  (:types robot location object)

  (:predicates
    (at-robot ?r - robot ?loc - location)
    (at-object ?o - object ?loc - location)
    (holding ?r - robot ?o - object)
  )

  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and (at-robot ?r ?from) (not (= ?from ?to)))
    :effect (and (not (at-robot ?r ?from)) (at-robot ?r ?to))
  )

  (:action pick_up
    :parameters (?r - robot ?o - object ?loc - location)
    :precondition (and (at-robot ?r ?loc) (at-object ?o ?loc))
    :effect (and (not (at-object ?o ?loc)) (holding ?r ?o))
  )

  (:action put_down
    :parameters (?r - robot ?o - object ?loc - location)
    :precondition (and (holding ?r ?o) (at-robot ?r ?loc))
    :effect (and (not (holding ?r ?o)) (at-object ?o ?loc))
  )
)"""

    problem = """(define (problem exercise0-problem-robot)
  (:domain exercise0)

  (:objects
    robot1 robot2 - robot
    loc1 loc2 loc3 - location
    box1 box2 - object
  )

  (:init
    (at-robot robot1 loc1)
    (at-robot robot2 loc2)
    (at-object box1 loc1)
    (at-object box2 loc3)
  )

  (:goal
    (and
      (at-robot robot1 loc2)
      (at-object box1 loc2)
      (at-object box2 loc1)
    )
  )
)"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "You are an IA PDDL planner that process PDDL domain and problem texts and generates plans for the goals of the problem."
            ),
            HumanMessagePromptTemplate.from_template(
                template=[
                    {
                        "type": "text",
                        "text": (
                            f"{domain}\n"
                            f"{problem}\n"
                            "Generate a plan to for the goals of the problem."
                        ),
                    },
                ]
            ),
        ]
    )

    chain = prompt | chat | StrOutputParser()

    initial_time = time.time()
    for text in chain.stream({}):
        tokens += 1
        print(text, end="", flush=True)
        if eval_time < 0:
            eval_time = time.time()

    print("", end="\n", flush=True)

    end_time = time.time()
    print(f"Time to eval: {eval_time - initial_time} s")
    print(f"Prediction speed: {tokens / (end_time - eval_time)} t/s")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
